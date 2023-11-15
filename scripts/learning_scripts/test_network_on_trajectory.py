from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
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
from kincalib.Calibration import RobotActualPoseCalulator
from kincalib.Motion.IkUtils import calculate_fk
import numpy as np

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

    sns.histplot(
        data=error_data,
        x="pos_error",
        ax=axes[0, 0],
        stat=stat,
        kde=True,
        bins=bins,
        hue="label",
    )
    sns.histplot(
        data=error_data,
        x="ori_error",
        ax=axes[0, 1],
        stat=stat,
        kde=True,
        bins=bins,
        hue="label",
    )
 


def plot_robot_error(
    measured_cp: np.ndarray, actual_cp: np.ndarray, corrupted_measured_cp: np.ndarray
):
    position_error1 = calculate_position_error(measured_cp, actual_cp)
    orientation_error1 = calculate_orientation_error(measured_cp, actual_cp)

    position_error2 = calculate_position_error(corrupted_measured_cp, actual_cp)
    orientation_error2 = calculate_orientation_error(corrupted_measured_cp, actual_cp)

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
        "measured-actual",
        position_error2,
        orientation_error2,
        "corrupted-actual",
    )

    plt.show()


def plot_correction_offset(
    experimental_data: RobotActualPoseCalulator, corrupted_measured_jp: np.ndarray
):
    sub_params = dict(
        top=0.88,
        bottom=0.06,
        left=0.07,
        right=0.95,
        hspace=0.62,
        wspace=0.31,
    )
    figsize = (7.56, 5.99)

    fig, ax = plt.subplots(6, 2, sharex=True, figsize=figsize)
    fig.subplots_adjust(**sub_params)

    correction_offset = experimental_data.actual_jp - experimental_data.measured_jp
    nn_offset = corrupted_measured_jp - experimental_data.measured_jp
    for i in range(6):
        ax[i, 0].plot(experimental_data.measured_jp[:, i])
        ax[i, 0].set_title(f"q{i+1}")
        ax[i, 1].plot(correction_offset[:, i], label="ground truth")
        ax[i, 1].plot(nn_offset[:, i], label="predicted_offset")
        ax[i, 1].set_title(f"correction offset q{i+1}")
    ax[0, 1].legend()
    [a.grid() for a in ax.flatten()]

    plt.show()


def load_robot_pose_cal(
    test_data_path: Path, hand_eye_path: Path
) -> RobotActualPoseCalulator:
    file_path = test_data_path
    hand_eye_file = hand_eye_path

    assert file_path.exists(), "File does not exist"
    assert hand_eye_file.exists(), "Hand eye file does not exist"

    log.info(f"Analyzing experiment {file_path.parent.name}")

    experimental_data = RobotActualPoseCalulator.load_from_file(
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
    ) -> NetworkNoiseGenerator:
        input_normalizer = Normalizer.from_json(input_normalizer_json)
        output_normalizer = Normalizer.from_json(output_normalizer_json)

        model = BestMLP2()
        model.load_state_dict(torch.load(weights_path))
        return cls(model, input_normalizer, output_normalizer)

    def batch_corrupt(self, measured_jp: np.ndarray) -> np.ndarray:
        """Corrupts a batch of measured_jp measurements with nn. Injected
        noise should minimize the error between actual_jp and measured_jp"""

        jp_norm = self.input_normalizer(measured_jp)
        jp_norm = torch.tensor(jp_norm.astype(np.float32))
        correction_offset = self.model(jp_norm).detach().numpy()
        correction_offset = self.output_normalizer.reverse(correction_offset)

        # Other way to do it that does not work
        # corrupted_actual2 = actual_jp
        # corrupted_actual2[:, 3:] += correction_offset[:, 3:]

        # correction_offset[:, :3] = 0  # Only apply the offset to last 3 joints
        corrupted_measured = measured_jp + correction_offset

        return corrupted_measured


def load_noise_generator(root: Path) -> NetworkNoiseGenerator:
    weights_path = root / "final_weights.pth"
    input_normalizer_path = root / "input_normalizer.json"
    output_normalizer_path = root / "output_normalizer.json"
    assert weights_path.exists(), f"Weights path {weights_path} does not exist"
    assert input_normalizer_path.exists(), f"{input_normalizer_path} does not exist"
    assert output_normalizer_path.exists(), f"{output_normalizer_path} does not exist"

    noise_generator = NetworkNoiseGenerator.create_from_files(
        weights_path, input_normalizer_path, output_normalizer_path
    )
    return noise_generator


def inject_errors(
    exp_data: RobotActualPoseCalulator, noise_generator: NetworkNoiseGenerator
):
    measured_jp = exp_data.measured_jp
    corrupted_measured_jp = noise_generator.batch_corrupt(measured_jp)
    corrupted_measured_cp = calculate_fk(corrupted_measured_jp)

    return corrupted_measured_cp, corrupted_measured_jp


def reduce_pose_error_with_nn():
    # fmt:off
    test_data_path = "./data/experiments/data_collection3/combined/record_001_3.csv"
    hand_eye_path = "./data/experiments/data_collection3/combined/hand_eye_calib.json"

    model_path = "./outputs_hydra/train_test_simple_net_20231114_223751" 
    # fmt:on

    test_data_path = Path(test_data_path)
    hand_eye_path = Path(hand_eye_path)
    model_path = Path(model_path)

    experimental_data = load_robot_pose_cal(test_data_path, hand_eye_path)
    noise_generator = load_noise_generator(model_path)

    corrupted_measured_cp, corrupted_measured_jp = inject_errors(
        experimental_data, noise_generator
    )

    plot_robot_error(
        experimental_data.measured_cp,
        experimental_data.actual_cp,
        corrupted_measured_cp,
    )
    plot_correction_offset(experimental_data, corrupted_measured_jp)


if __name__ == "__main__":
    reduce_pose_error_with_nn()
