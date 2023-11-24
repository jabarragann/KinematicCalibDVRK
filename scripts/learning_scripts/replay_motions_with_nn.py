from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple
import hydra
import sys

import pandas as pd
from kincalib.Calibration.RobotPosesContainer import RobotPosesContainer
from replay_motions_with_nn_confs.structured_confs import (
    AppConfig,
    register_struct_configs,
)
from kincalib.utils.Logger import Logger
from omegaconf import OmegaConf
from test_network_on_trajectory import NetworkNoiseGenerator, load_robot_pose_cal
from kincalib.Motion.IkUtils import calculate_fk
from kincalib.utils import calculate_orientation_error, calculate_position_error
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# log = logging.getLogger(__name__)
log = Logger(__name__).log

register_struct_configs()


@dataclass
class ValidateConfig:
    cfg: AppConfig

    def __post_init__(self):
        self.validate_nn_conf()
        self.validate_rosbag_path()
        self.validate_test_data()
        log.info("Configuration is valid")

    def validate_test_data(self):
        assert (
            self.cfg.test_data_path.exists()
        ), f"Test data path {self.cfg.test_data_path} does not exist"
        assert (
            self.cfg.hand_eye_path.exists()
        ), f"Hand eye path {self.cfg.hand_eye_path} does not exist"

    def validate_rosbag_path(self):
        assert (
            self.cfg.rosbag_path.exists()
        ), f"File {self.cfg.rosbag_path} does not exist"

    def validate_nn_conf(self):
        self.is_neural_net_trained_with(
            self.cfg.nn_measured_setpoint_path, "measured-setpoint"
        )
        self.is_neural_net_trained_with(
            self.cfg.nn_actual_measured_path, "actual-measured"
        )

    def is_neural_net_trained_with(self, nn_path: Path, dataset_type: str):
        with open(nn_path / "dataset_info.json", "r") as file:
            cfg_dict = json.load(file)
            type_in_config = cfg_dict["dataset_type"]
        assert (
            dataset_type == type_in_config
        ), f"Neural net in {nn_path} was trained with {type_in_config} should have been trained with {dataset_type} dataset."


def load_noise_generators(
    cfg: AppConfig,
) -> Tuple[NetworkNoiseGenerator, NetworkNoiseGenerator]:
    nn1_measured_setpoint = NetworkNoiseGenerator.create_from_files(
        cfg.nn_measured_setpoint_path / "final_weights.pth",
        cfg.nn_measured_setpoint_path / "input_normalizer.json",
        cfg.nn_measured_setpoint_path / "output_normalizer.json",
    )
    nn2_actual_measured = NetworkNoiseGenerator.create_from_files(
        cfg.nn_actual_measured_path / "final_weights.pth",
        cfg.nn_actual_measured_path / "input_normalizer.json",
        cfg.nn_actual_measured_path / "output_normalizer.json",
    )
    return nn1_measured_setpoint, nn2_actual_measured


def generate_error_hist(
    real_robot_poses: RobotPosesContainer, simulated_robot_poses: RobotPosesContainer
):
    simulated_robot_error_df = simulated_robot_poses.convert_to_error_dataframe()
    real_robot_error_df = real_robot_poses.convert_to_error_dataframe()
    error_data = pd.concat([simulated_robot_error_df, real_robot_error_df])

    fig, axes = plt.subplots(2, 2)
    stat = "proportion"
    bins = 50

    # fmt: off
    # approx_actual_cp-approx_measured_cp <==> actual_cp-measured_cp
    # approx_measured_cp-setpoint_cp <==> measured_cp-setpoint_cp
    sns.histplot( data=error_data, x="pos_error_actual_measured", ax=axes[0, 0], stat=stat, kde=True, bins=bins, hue="label",)
    sns.histplot( data=error_data, x="ori_error_actual_measured", ax=axes[0, 1], stat=stat, kde=True, bins=bins, hue="label",)
    sns.histplot( data=error_data, x="pos_error_measured_setpoint", ax=axes[1, 0], stat=stat, kde=True, bins=bins, hue="label",)
    sns.histplot( data=error_data, x="ori_error_measured_setpoint", ax=axes[1, 1], stat=stat, kde=True, bins=bins, hue="label",)
    # fmt: on

    plt.show()


def calculate_simulated_poses(
    real_robot_poses: RobotPosesContainer,
    nn1_measured_setpoint: NetworkNoiseGenerator,
    nn2_actual_measured: NetworkNoiseGenerator,
):
    approximated_measured_jp, alpha1 = nn1_measured_setpoint.corrupt_jp_batch(
        real_robot_poses.setpoint_jp
    )

    # approximated_measured_jp should be returned instead of ambf.measured_jp()
    # approximated_measured_jp = robot_poses.setpoint_jp + alpha1
    approximated_actual_jp, alpha2 = nn2_actual_measured.corrupt_jp_batch(
        approximated_measured_jp
    )
    # Corrupted setpoints are given to ambf.move_jp() command
    corrupted_setpoints2 = real_robot_poses.setpoint_jp + alpha1 + alpha2

    # Assuming no controller error on the simulated robot
    # approximated_actual_jp = corrupted_setpoints2. Here they should be the same.
    assert np.all(
        np.isclose(approximated_actual_jp, corrupted_setpoints2)
    ), "Error with nn logic"

    simulated_robot_poses = RobotPosesContainer.create_from_jp_values(
        "simulated_robot",
        real_robot_poses.index_array,
        real_robot_poses.setpoint_jp,
        approximated_measured_jp,
        approximated_actual_jp,
    )

    return simulated_robot_poses


def replay_motions(cfg: AppConfig, simulated_robot_poses: RobotPosesContainer):
    from kincalib.Motion.ReplayDevice import create_psm_handle
    from kincalib.Motion.TrajectoryPlayer import TrajectoryPlayer, Trajectory

    class SimpleRec:
        def __init__(self, arm):
            self.arm = arm
            self.data = []
            self.data_idx = []

        def collect_data(self, index):
            self.data_idx.append(index)
            self.data.append(self.arm.measured_jp())

        def get_data(self) -> np.ndarray:
            return np.array(self.data)

    total_idx = simulated_robot_poses.index_array.shape[0]
    trajectory = Trajectory.from_setpoint_array(
        simulated_robot_poses.actual_jp[:total_idx, :]
    )
    arm = create_psm_handle("PSM2", type="ambf", expected_interval=0.01)
    data_recorder = SimpleRec(arm)

    trajectory_player = TrajectoryPlayer(
        replay_device=arm,
        trajectory=trajectory,
        before_motion_loop_cb=[],
        after_motion_cb=[data_recorder.collect_data],
    )
    trajectory_player.replay_trajectory(execute_cb=True)

    simulated_robot_poses2 = RobotPosesContainer.create_from_jp_values(
        "simulated_robot",
        simulated_robot_poses.index_array[:total_idx],
        simulated_robot_poses.setpoint_jp[:total_idx],
        simulated_robot_poses.measured_jp[:total_idx],
        data_recorder.get_data(),
    )
    return simulated_robot_poses2


@hydra.main(
    version_base=None,
    config_path="./replay_motions_with_nn_confs",
    config_name="replay_motions_with_nn",
)
def main(cfg: AppConfig):
    print(OmegaConf.to_yaml(cfg))
    ValidateConfig(cfg)
    log.info(type(cfg.test_data_path))

    nn1_measured_setpoint, nn2_actual_measured = load_noise_generators(cfg)
    real_robot_poses = load_robot_pose_cal(cfg.test_data_path, cfg.hand_eye_path)

    simulated_robot_poses1 = calculate_simulated_poses(
        real_robot_poses, nn1_measured_setpoint, nn2_actual_measured
    )

    generate_error_hist(real_robot_poses, simulated_robot_poses1)

    # simulated_robot_poses2 = replay_motions(cfg, simulated_robot_poses1)

    # # fmt: off
    # # Disable filter by raising the thresholds
    # real_robot_poses.filter_and_save_to_record(cfg.output_path / "real_robot_poses.csv", pos_error_threshold=800, orientation_error_threshold=1000)
    # simulated_robot_poses1.filter_and_save_to_record(cfg.output_path / "simulated_robot_poses1.csv", pos_error_threshold=800, orientation_error_threshold=1000)
    # simulated_robot_poses2.filter_and_save_to_record(cfg.output_path / "simulated_robot_poses2.csv", pos_error_threshold=800, orientation_error_threshold=1000)
    # # fmt: on


if __name__ == "__main__":
    main()
