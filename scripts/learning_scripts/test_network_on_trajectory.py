from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import click
import pandas as pd

import torch
from kincalib.Learning.Dataset import Normalizer
from kincalib.Learning.Models import BestMLP2
from kincalib.utils.Logger import Logger
from kincalib.utils import (
    create_cartesian_error_histogram,
    create_cartesian_error_lineplot,
)
from kincalib.utils import calculate_orientation_error, calculate_position_error
import matplotlib.pyplot as plt
import seaborn as sns
from kincalib.Calibration import RobotPosesContainer
from kincalib.Motion.IkUtils import batch_calculate_fk
import numpy as np
from omegaconf import OmegaConf
from kincalib.Kinematics import DvrkPsmKin_SRC

# Structured config files for the training script
from train_test_simple_net_confs.structured_confs import ExperimentConfig

log = Logger(__name__).log


def plot_histograms(
    pos_err1,
    ori_err1,
    label_err1,
    pos_err2,
    ori_err2,
    label_err2,
    stat="proportion",
    bins=50,
):
    data_dict = dict(pos_error=pos_err1, ori_error=ori_err1, label=label_err1)
    error_data1 = pd.DataFrame(data_dict)
    data_dict = dict(pos_error=pos_err2, ori_error=ori_err2, label=label_err2)
    error_data2 = pd.DataFrame(data_dict)

    error_data = pd.concat([error_data1, error_data2])

    fig, axes = plt.subplots(1, 2)
    axes = np.expand_dims(axes, axis=0)

    # fmt: off
    sns.histplot(data=error_data, x="pos_error", ax=axes[0, 0], stat=stat, kde=True, bins=bins, hue="label",)
    sns.histplot(data=error_data, x="ori_error", ax=axes[0, 1], stat=stat, kde=True, bins=bins, hue="label",)
    # fmt: on


def plot_cartesian_errors(
    poses1: np.ndarray,
    poses2: np.ndarray,
    poses2_approximated: np.ndarray,
    poses1_name: str,
    poses2_name: str,
):
    """Plots the cartesian error between two sets of poses Poses1 and poses2
    are ground truth poses from measurements.  Poses2_approximated are
    calculated from poses1 using the neural network and should approximate
    poses2. Only two different settings are considered: 1) poses1 = setpoint_cp,
    poses2 = measured_cp, and 2) poses1 = measured_cp, poses2 = actual_cp

    parameters
    ----------

    poses1: np.ndarray
        Ground truth poses from measurements
    poses2: np.ndarray
        Ground truth poses from measurements
    poses2_approximated: np.ndarray
        Approximated poses from poses1 using the neural network

    """
    position_error1 = calculate_position_error(poses1, poses2)
    orientation_error1 = calculate_orientation_error(poses1, poses2)

    position_error2 = calculate_position_error(poses2_approximated, poses2)
    orientation_error2 = calculate_orientation_error(poses2_approximated, poses2)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].set_title("Error between measured and actual")
    create_cartesian_error_lineplot(
        position_error1, orientation_error1, axes[0, 0], axes[1, 0]
    )
    axes[0, 1].set_title("Error between corrupted and actual")
    create_cartesian_error_lineplot(
        position_error2, orientation_error2, axes[0, 1], axes[1, 1]
    )

    plot_histograms(
        position_error1,
        orientation_error1,
        f"{poses2_name}-{poses1_name}",
        position_error2,
        orientation_error2,
        f"{poses2_name}-{poses2_name}_approx",
    )

    plt.show()


def plot_correction_offset(
    jp1: np.ndarray, jp2: np.ndarray, jp2_approximate: np.ndarray
):
    sub_params = dict(
        top=0.88, bottom=0.06, left=0.07, right=0.95, hspace=0.62, wspace=0.31
    )
    figsize = (7.56, 5.99)

    fig, ax = plt.subplots(6, 2, sharex=True, figsize=figsize)
    fig.subplots_adjust(**sub_params)

    correction_offset = jp2 - jp1

    # jp2_approx = jp1 + nn_offset = jp1 + (jp2 - jp1) = jp2
    nn_offset = jp2_approximate - jp1
    for i in range(6):
        ax[i, 0].plot(jp1[:, i])
        ax[i, 0].set_title(f"q{i+1}")
        ax[i, 1].plot(correction_offset[:, i], label="ground truth")
        ax[i, 1].plot(nn_offset[:, i], label="predicted_offset")
        ax[i, 1].set_title(f"correction offset q{i+1}")
    ax[0, 1].legend()
    [a.grid() for a in ax.flatten()]

    plt.show()


def load_robot_poses(test_data_path: Path, hand_eye_path: Path) -> RobotPosesContainer:
    file_path = test_data_path
    hand_eye_file = hand_eye_path

    assert file_path.exists(), "File does not exist"
    assert hand_eye_file.exists(), "Hand eye file does not exist"

    log.info(f"Analyzing experiment {file_path.parent.name}")

    experimental_data = RobotPosesContainer.create_from_real_measurements(
        file_path=file_path, hand_eye_file=hand_eye_file
    )

    return experimental_data


@dataclass
class NetworkNoiseGenerator:
    """Corrupts a batch of jp measurements with nn. Injected
    noise should minimize the error between actual_jp and measured_jp"""

    model: torch.nn.Module
    input_normalizer: Normalizer
    output_normalizer: Normalizer

    @classmethod
    def create_from_files(
        cls: NetworkNoiseGenerator,
        weights_path: Path,
        input_normalizer_json: Path,
        output_normalizer_json: Path,
        input_features: int,
    ) -> NetworkNoiseGenerator:
        input_normalizer = Normalizer.from_json(input_normalizer_json)
        output_normalizer = Normalizer.from_json(output_normalizer_json)

        model = BestMLP2(input_features)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        return cls(model, input_normalizer, output_normalizer)

    def corrupt_jp_batch(
        self, input_jp: np.ndarray, prev_measured: np.ndarray, cfg: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        if cfg["include_prev_measured"]:
            complete_input = np.hstack((input_jp, prev_measured))
        else:
            complete_input = input_jp

        jp_norm = self.input_normalizer(complete_input)
        jp_norm = torch.tensor(jp_norm.astype(np.float32))
        offset = self.model(jp_norm).detach().numpy()
        offset = self.output_normalizer.reverse(offset)

        # Other way to do it that does not work
        # corrupted_actual2 = actual_jp
        # corrupted_actual2[:, 3:] += correction_offset[:, 3:]

        # correction_offset[:, :3] = 0  # Only apply the offset to last 3 joints
        corrupted_jp = input_jp + offset

        return corrupted_jp, offset

    def inject_errors(
        self, poses1_jp: np.ndarray, prev_measured: np.ndarray, cfg: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Corrupts a batch of jp measurements with nn. Injected
        noise should minimize the error between actual_jp/measured_jp
        or measured_jp/setpoint_jp depending on the used network."""

        poses2_jp_approximate, offset = self.corrupt_jp_batch(
            poses1_jp, prev_measured, cfg
        )
        kin_model = DvrkPsmKin_SRC("classic") 
        poses2_cp_approximate = batch_calculate_fk(poses2_jp_approximate, kin_model)

        return poses2_cp_approximate, poses2_jp_approximate


def load_noise_generator(root: Path, input_features: int) -> NetworkNoiseGenerator:
    weights_path = root / "final_weights.pth"
    input_normalizer_path = root / "input_normalizer.json"
    output_normalizer_path = root / "output_normalizer.json"
    assert weights_path.exists(), f"Weights path {weights_path} does not exist"
    assert input_normalizer_path.exists(), f"{input_normalizer_path} does not exist"
    assert output_normalizer_path.exists(), f"{output_normalizer_path} does not exist"

    noise_generator = NetworkNoiseGenerator.create_from_files(
        weights_path, input_normalizer_path, output_normalizer_path, input_features
    )
    return noise_generator


def load_dataset_config(model_path: Path) -> Dict[str, Any]:
    dataset_info_path = model_path / "dataset_info.json"
    assert (
        dataset_info_path.exists()
    ), f"Dataset info path {dataset_info_path} does not exist"
    with open(dataset_info_path, "r") as file:
        cfg_dict = json.load(file)

    return cfg_dict


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="outputs_hydra/train_test_simple_net_20231129_200704",
)
@click.option(
    "--test_data_name",
    type=str,
    default="raw_sensor_rosbag_09_traj3.csv",
    help="Name of trajectory to use. This file needs to be in the datapath",
)
@click.option(
    "--output_path",
    type=click.Path(path_type=Path),
    default=None,
)
def reduce_pose_error_with_nn(model_path, test_data_name, output_path):
    # Load config from experiment
    log.info(f"Loading model from {model_path}")
    cfg: ExperimentConfig  # For Duck typing
    cfg = OmegaConf.load(model_path / ".hydra/config.yaml")
    cfg_hydra = OmegaConf.load(model_path / ".hydra/hydra.yaml")
    cfg.output_path = cfg_hydra.hydra.runtime.output_dir

    # Load data
    test_data_path = Path(cfg.path_config.datapath) / test_data_name
    hand_eye_path = Path(cfg.path_config.handeye_path)
    model_path = Path(model_path)
    log.info(f"Hand eye calib path: {hand_eye_path}")
    log.info(f"Load trajectory {test_data_path}")

    cfg_dict = load_dataset_config(model_path)
    dataset_type = cfg_dict["dataset_type"]
    include_prev_measured = cfg_dict["include_prev_measured"]
    input_features = 12 if include_prev_measured else 6

    log.info(f"Loading a model with dataset cfg:\n {cfg_dict}")

    experimental_data = load_robot_poses(test_data_path, hand_eye_path)
    noise_generator = load_noise_generator(model_path, input_features)

    if dataset_type == "measured-setpoint":
        poses1_cp = experimental_data.setpoint_cp
        poses1_jp = experimental_data.setpoint_jp
        poses1_name = "setpoint"
        poses2_cp = experimental_data.measured_cp
        poses2_jp = experimental_data.measured_jp
        poses2_name = "measured"
    elif dataset_type == "actual-measured":
        poses1_cp = experimental_data.measured_cp
        poses1_jp = experimental_data.measured_jp
        poses1_name = "measured"
        poses2_cp = experimental_data.actual_cp
        poses2_jp = experimental_data.actual_jp
        poses2_name = "actual"
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    tminus1_measured_jp = experimental_data.prev_measured_jp["measured_jp_tminus1"]

    poses2_cp_approximate, poses2_jp_approximate = noise_generator.inject_errors(
        poses1_jp, tminus1_measured_jp, cfg_dict
    )

    plot_cartesian_errors(
        poses1_cp, poses2_cp, poses2_cp_approximate, poses1_name, poses2_name
    )
    plot_correction_offset(poses1_jp, poses2_jp, poses2_jp_approximate)

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / f"poses1_jp_{dataset_type}.npy", poses1_jp)
        np.save(output_path / f"poses2_jp_{dataset_type}.npy", poses2_jp)
        np.save(
            output_path / f"poses2_jp_pred_{dataset_type}.npy", poses2_jp_approximate
        )
        np.save(output_path / f"poses1_cp_{dataset_type}.npy", poses1_cp)
        np.save(output_path / f"poses2_cp_{dataset_type}.npy", poses2_cp)


if __name__ == "__main__":
    reduce_pose_error_with_nn()
