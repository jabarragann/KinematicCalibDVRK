from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

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


def plot_robot_error(measured_cp: np.ndarray, corrupted_actual_cp: np.ndarray):
    position_error = calculate_position_error(measured_cp, corrupted_actual_cp)
    orientation_error = calculate_orientation_error(measured_cp, corrupted_actual_cp)

    fig, axes = create_cartesian_error_lineplot(position_error, orientation_error)
    fig, axes = create_cartesian_error_histogram(position_error, orientation_error)

    plt.show()


def plot_correction_offset(
    experimental_data: RobotActualPoseCalulator, corrupted_actual_jp: np.ndarray
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

    correction_offset = experimental_data.measured_jp - experimental_data.actual_jp
    nn_offset = experimental_data.measured_jp - corrupted_actual_jp
    for i in range(6):
        ax[i, 0].plot(experimental_data.measured_jp[:, i])
        ax[i, 0].set_title(f"q{i+1}")
        ax[i, 1].plot(correction_offset[:, i], label="ground truth")
        ax[i, 1].plot(nn_offset[:, i], label="predicted_offset")
        ax[i, 1].set_title(f"correction offset q{i+1}")
    ax[0, 1].legend()
    [a.grid() for a in ax.flatten()]

    plt.show()


def load_robot_pose_cal(exp_root: Path) -> RobotActualPoseCalulator:
    file_path = exp_root / "combined_data.csv"
    hand_eye_file = exp_root / "hand_eye_calib.json"

    assert file_path.exists(), "File does not exist"
    assert hand_eye_file.exists(), "Hand eye file does not exist"

    log.info(f"Analyzing experiment {file_path.parent.name}")

    experimental_data = RobotActualPoseCalulator.load_from_file(
        file_path=file_path, hand_eye_file=hand_eye_file
    )

    return experimental_data


@dataclass
class NetworkNoiseGenerator:
    """Corrupts a batch of actual_jp measurements with nn. Injected
    noise should minimize the error between actual_jp and measured_jp"""

    model: torch.nn.Module
    normalizer: Normalizer

    @classmethod
    def create_from_files(
        cls: NetworkNoiseGenerator, weights_path: Path, normalizer_json: Path
    ) -> NetworkNoiseGenerator:
        normalizer = Normalizer.from_json(normalizer_json)
        model = BestMLP2()
        model.load_state_dict(torch.load(weights_path))
        return cls(model, normalizer)

    def batch_corrupt(self, actual_jp: np.ndarray) -> np.ndarray:
        """Corrupts a batch of actual_jp measurements with nn. Injected
        noise should minimize the error between actual_jp and measured_jp"""

        jp_norm = self.normalizer(actual_jp)
        jp_norm = torch.tensor(jp_norm.astype(np.float32))
        correction_offset = self.model(jp_norm).detach().numpy()

        corrupted_actual = actual_jp + correction_offset
        return corrupted_actual


def load_noise_generator(root: Path) -> NetworkNoiseGenerator:
    weights_path = root / "final_weights.pth"
    normalizer_path = root / "normalizer.json"
    assert weights_path.exists(), f"Weights path {weights_path} does not exist"
    assert normalizer_path.exists(), f"Normalizer path {normalizer_path} does not exist"

    noise_generator = NetworkNoiseGenerator.create_from_files(
        weights_path, normalizer_path
    )
    return noise_generator


def inject_errors(
    exp_data: RobotActualPoseCalulator, noise_generator: NetworkNoiseGenerator
):
    actual_jp = exp_data.actual_jp
    corrupted_actual_jp = noise_generator.batch_corrupt(actual_jp)
    corrupted_actual_cp = calculate_fk(corrupted_actual_jp)

    return corrupted_actual_cp, corrupted_actual_jp


def reduce_pose_error_with_nn():
    # fmt:off
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-24-30"
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-28-58"
    exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-33-24"

    # # exp_root = "./data/experiments/data_collection1/08-11-2023-19-23-55"
    # # exp_root = "./data/experiments/data_collection1/08-11-2023-19-33-54"
    # exp_root = "./data/experiments/data_collection1/08-11-2023-19-52-14"

    model_path = "./outputs_hydra/train_test_simple_net_20231112_191433" 
    # fmt:on

    exp_root = Path(exp_root)
    model_path = Path(model_path)

    experimental_data = load_robot_pose_cal(exp_root)
    noise_generator = load_noise_generator(model_path)

    corrupted_actual_cp, corrupted_actual_jp = inject_errors(
        experimental_data, noise_generator
    )

    plot_robot_error(experimental_data.T_RG, corrupted_actual_cp)
    plot_correction_offset(experimental_data, corrupted_actual_jp)


if __name__ == "__main__":
    reduce_pose_error_with_nn()
