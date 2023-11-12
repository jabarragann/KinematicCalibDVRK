from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import kincalib
from kincalib.Record.DataRecorder import DataReaderFromCSV, DataRecorder
from kincalib.Calibration.HandEyeCalibration import HandEyeBatchProcessing
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.Logger import Logger
from kincalib.utils import calculate_orientation_error, calculate_position_error
import kincalib.Record as records
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from kincalib.Calibration import RobotActualPoseCalulator, HandEyeBatchProcessing

log = Logger(__name__).log


def plot_robot_error(experimental_data: RobotActualPoseCalulator):
    position_error = experimental_data.position_error
    orientation_error = experimental_data.orientation_error
    error_data = experimental_data.convert_to_dataframe()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax = np.expand_dims(ax, axis=0)

    ax[0, 0].plot(position_error)
    ax[0, 0].set_ylabel("Position error (mm)")
    ax[0, 1].plot(orientation_error)
    ax[0, 1].set_ylabel("Orientation error (deg)")

    # for i in range(3):
    #     ax[i+1,0].plot(measured_jp[:,3+i])
    #     ax[i+1,1].plot(measured_jp[:,3+i])

    [a.grid() for a in ax.flatten()]
    # major_ticks = np.arange(0,350,25)
    # [a.set_xticks(major_ticks) for a in ax.flatten()]

    # correlation plot
    fig, ax = plt.subplots(3, 2)
    sns.scatterplot(x="q4", y="pos_error", data=error_data, ax=ax[0, 0])
    sns.scatterplot(x="q5", y="pos_error", data=error_data, ax=ax[1, 0])
    sns.scatterplot(x="q6", y="pos_error", data=error_data, ax=ax[2, 0])

    sns.scatterplot(x="q4", y="orientation_error", data=error_data, ax=ax[0, 1])
    sns.scatterplot(x="q5", y="orientation_error", data=error_data, ax=ax[1, 1])
    sns.scatterplot(x="q6", y="orientation_error", data=error_data, ax=ax[2, 1])

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f"Error distribution (N={error_data.shape[0]})")
    # ax = np.expand_dims(ax, axis=0)
    stat = "proportion"
    sns.histplot(data=error_data, x="pos_error", ax=ax[0, 0], stat=stat, kde=True)
    sns.histplot(
        data=error_data, x="orientation_error", ax=ax[0, 1], stat=stat, kde=True
    )
    sns.histplot(
        data=error_data,
        x="pos_error",
        ax=ax[1, 0],
        stat=stat,
        kde=True,
        cumulative=True,
    )
    sns.histplot(
        data=error_data,
        x="orientation_error",
        ax=ax[1, 1],
        stat=stat,
        kde=True,
        cumulative=True,
    )
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

    experimental_data.save_to_record(output_path=exp_root)
    plot_robot_error(experimental_data)
    plot_correction_offset(experimental_data)


if __name__ == "__main__":
    analyze_robot_error()
