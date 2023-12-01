from __future__ import annotations
from pathlib import Path
import pandas as pd
import kincalib
from kincalib.Record.DataRecorder import SensorsDataReader, RealDataRecorder
from kincalib.utils.Logger import Logger
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from kincalib.utils import calculate_orientation_error, calculate_position_error


log = Logger(__name__).log


@dataclass
class ExperimentalData:
    setpoint_jp: np.ndarray
    setpoint_cp: np.ndarray
    measured_jp: np.ndarray
    measured_cp: np.ndarray

    def __post_init__(self):
        pass

    @classmethod
    def load_from_file(cls: ExperimentalData, file_path: Path):
        assert file_path.exists(), f"File {file_path} does not exist"

        record_dict = RealDataRecorder.create_records()
        data_dict = SensorsDataReader(file_path, record_dict).data_dict

        return ExperimentalData(
            setpoint_jp=data_dict["setpoint_jp"],
            setpoint_cp=data_dict["setpoint_cp"],
            measured_jp=data_dict["measured_jp"],
            measured_cp=data_dict["measured_cp"],
        )


def plot_measured_jp(robot_data: ExperimentalData, sim_data: ExperimentalData):
    fig, ax = plt.subplots(6, 2, sharex=True)

    for i in range(6):
        ax[i, 0].plot(robot_data.measured_jp[:, i], label="real")
        ax[i, 0].plot(sim_data.measured_jp[:, i], label="sim")
        error = robot_data.measured_jp[:, i] - sim_data.measured_jp[:, i]
        ax[i, 1].plot(error, label="error")

    ax[0, 0].legend()
    [a.grid() for a in ax.flatten()]


def plot_cartesian_errors(robot_data: ExperimentalData, sim_data: ExperimentalData):
    cartesian_err = calculate_position_error(
        T_RG=robot_data.measured_cp, T_RG_actual=sim_data.measured_cp
    )
    rot_err = calculate_orientation_error(
        T_RG=robot_data.measured_cp, T_RG_actual=sim_data.measured_cp
    )

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Difference between real and simulated robot")
    ax[0].plot(cartesian_err, label="cartesian error")
    ax[0].set_title("Cartesian error (mm)")
    ax[1].plot(rot_err, label="rot error")
    ax[1].set_title("Rot error (deg)")

    [a.grid() for a in ax.flatten()]


def plot_controller_errors(robot_data: ExperimentalData, sim_data: ExperimentalData):
    real_controller_err_p = calculate_position_error(
        T_RG=robot_data.measured_cp, T_RG_actual=robot_data.setpoint_cp
    )
    real_controller_err_r = calculate_orientation_error(
        T_RG=robot_data.measured_cp, T_RG_actual=robot_data.setpoint_cp
    )

    sim_controller_err_p = calculate_position_error(
        T_RG=sim_data.measured_cp, T_RG_actual=sim_data.setpoint_cp
    )
    sim_controller_err_r = calculate_orientation_error(
        T_RG=sim_data.measured_cp, T_RG_actual=sim_data.setpoint_cp
    )

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Difference between real and simulated robot")
    ax[0].plot(real_controller_err_p, label="real")
    ax[0].plot(sim_controller_err_p, label="sim")
    ax[0].set_title("Cartesian error (mm)")
    ax[1].plot(real_controller_err_r, label="real")
    ax[1].plot(sim_controller_err_r, label="sim")
    ax[1].set_title("Rot error (deg)")

    [a.grid() for a in ax.flatten()]
    ax[0].legend()


def analyze_robot_error():
    # fmt: off
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-24-30"
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-28-58"
    real_robot_root = "./data/experiments/data_collection3/combined/bag1_traj_2.csv"
    sim_robot_root = "./data/experiments/simulated_robot/19-11-2023-10-35-28/record_001.csv"
    # fmt: on

    real_robot_root = Path(real_robot_root)
    sim_robot_root = Path(sim_robot_root)

    robot_data = ExperimentalData.load_from_file(file_path=real_robot_root)
    sim_data = ExperimentalData.load_from_file(file_path=sim_robot_root)

    # plot_measured_jp(robot_data, sim_data)
    plot_cartesian_errors(robot_data, sim_data)
    plot_controller_errors(robot_data, sim_data)

    plt.show()


if __name__ == "__main__":
    analyze_robot_error()
