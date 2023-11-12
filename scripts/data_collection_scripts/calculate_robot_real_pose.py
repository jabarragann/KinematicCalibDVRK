from __future__ import annotations
from pathlib import Path
from kincalib.utils.Logger import Logger
from kincalib.utils import (
    create_cartesian_error_histogram,
    create_cartesian_error_lineplot,
)
import matplotlib.pyplot as plt
import seaborn as sns
from kincalib.Calibration import RobotActualPoseCalulator

log = Logger(__name__).log


def plot_robot_error(experimental_data: RobotActualPoseCalulator):
    position_error = experimental_data.position_error
    orientation_error = experimental_data.orientation_error
    error_data = experimental_data.convert_to_dataframe()

    fig, axes = create_cartesian_error_lineplot(position_error, orientation_error)

    # correlation plot
    fig, ax = plt.subplots(3, 2)
    sns.scatterplot(x="q4", y="pos_error", data=error_data, ax=ax[0, 0])
    sns.scatterplot(x="q5", y="pos_error", data=error_data, ax=ax[1, 0])
    sns.scatterplot(x="q6", y="pos_error", data=error_data, ax=ax[2, 0])

    sns.scatterplot(x="q4", y="orientation_error", data=error_data, ax=ax[0, 1])
    sns.scatterplot(x="q5", y="orientation_error", data=error_data, ax=ax[1, 1])
    sns.scatterplot(x="q6", y="orientation_error", data=error_data, ax=ax[2, 1])

    fig, axes = create_cartesian_error_histogram(position_error, orientation_error)

    plt.show()


def plot_correction_offset(experimental_data: RobotActualPoseCalulator):
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
    for i in range(6):
        ax[i, 0].plot(experimental_data.measured_jp[:, i])
        ax[i, 0].set_title(f"q{i+1}")
        ax[i, 1].plot(correction_offset[:, i])
        ax[i, 1].set_title(f"correction offset q{i+1}")

    plt.show()


def analyze_robot_error():
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-24-30"
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-28-58"
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-33-24"

    # exp_root = "./data/experiments/data_collection1/08-11-2023-19-23-55"
    # exp_root = "./data/experiments/data_collection1/08-11-2023-19-33-54"
    exp_root = "./data/experiments/data_collection1/08-11-2023-19-52-14"

    exp_root = Path(exp_root)

    file_path = exp_root / "combined_data.csv"
    hand_eye_file = exp_root / "hand_eye_calib.json"

    assert file_path.exists(), "File does not exist"
    assert hand_eye_file.exists(), "Hand eye file does not exist"

    log.info(f"Analyzing experiment {file_path.parent.name}")

    experimental_data = RobotActualPoseCalulator.load_from_file(
        file_path=file_path, hand_eye_file=hand_eye_file
    )

    experimental_data.filter_and_save_to_record(output_path=exp_root)
    plot_robot_error(experimental_data)
    plot_correction_offset(experimental_data)


if __name__ == "__main__":
    analyze_robot_error()
