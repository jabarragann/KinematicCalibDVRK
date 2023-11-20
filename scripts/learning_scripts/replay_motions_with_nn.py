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


def calculate_simulated_poses(
    robot_poses: RobotPosesContainer,
    nn1_measured_setpoint: NetworkNoiseGenerator,
    nn2_actual_measured: NetworkNoiseGenerator,
):
    approximated_measured_jp, alpha1 = nn1_measured_setpoint.corrupt_jp_batch(
        robot_poses.setpoint_jp
    )

    # approximated_measured_jp should be the returned instead of ambf.measured_jp()
    # approximated_measured_jp = robot_poses.setpoint_jp + alpha1
    corrupted_setpoints1, alpha2 = nn2_actual_measured.corrupt_jp_batch(
        approximated_measured_jp
    )
    # Corrupted setpoints are given to ambf.move_jp() command
    corrupted_setpoints2 = robot_poses.setpoint_jp + alpha1 + alpha2
    # Assuming no controller error on the simulated robot
    approximated_actual_jp = corrupted_setpoints2

    approximated_actual_cp = calculate_fk(approximated_actual_jp)
    approximated_measured_cp = calculate_fk(approximated_measured_jp)

    simulated_pos_error_actual_measured = calculate_position_error(
        approximated_measured_cp, approximated_actual_cp
    )
    simulated_ori_error_actual_measured = calculate_orientation_error(
        approximated_measured_cp, approximated_actual_cp
    )

    simulated_pos_error_measured_setpoint = calculate_position_error(
        robot_poses.setpoint_cp, approximated_measured_cp
    )
    simulated_ori_error_measured_setpoint = calculate_orientation_error(
        robot_poses.setpoint_cp, approximated_actual_cp
    )

    data_dict = dict(
        pos_error_actual_measured=simulated_pos_error_actual_measured,
        ori_error_actual_measured=simulated_ori_error_actual_measured,
        pos_error_measured_setpoint=simulated_pos_error_measured_setpoint,
        ori_error_measured_setpoint=simulated_ori_error_measured_setpoint,
        label="simulated-robot",
    )
    error_data1 = pd.DataFrame(data_dict)
    # data_dict = dict(
    #     pos_error_actual_measured=robot_poses.position_error_actual_measured,
    #     ori_error_actual_measured=robot_poses.orientation_error_actual_measured,
    #     pos_error_measured_setpoint=robot_poses.position_error_measured_setpoint,
    #     ori_error_measured_setpoint=robot_poses.orientation_error_measured_setpoint,
    #     label="real-robot",
    # )
    # error_data2 = pd.DataFrame(data_dict)
    real_robot_error_df = robot_poses.convert_to_error_dataframe()
    error_data = pd.concat([error_data1, real_robot_error_df])

    fig, axes = plt.subplots(2, 2)
    # axes = np.expand_dims(axes, axis=0)

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
    robot_pose_data = load_robot_pose_cal(cfg.test_data_path, cfg.hand_eye_path)

    calculate_simulated_poses(
        robot_pose_data, nn1_measured_setpoint, nn2_actual_measured
    )


if __name__ == "__main__":
    main()
