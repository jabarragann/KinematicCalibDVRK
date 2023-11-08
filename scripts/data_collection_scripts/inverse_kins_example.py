from pathlib import Path
from typing import Dict, List
from matplotlib import pyplot as plt

import pandas as pd
from kincalib.Record.DataRecorder import DataReaderFromCSV, DataRecorder
from kincalib.Calibration.HandEyeCalibration import Batch_Processing
from kincalib.utils.Logger import Logger
from kincalib.Motion.DvrkKin import DvrkPsmKin
from surgical_robotics_challenge.kinematics.psmIK import compute_FK, compute_IK, convert_mat_to_frame
import surgical_robotics_challenge
import json
import numpy as np

log = Logger(__name__).log

def plot_joints(data_dict: Dict[str, np.ndarray]):
    cal_jp = calculate_ik(data_dict)

    sub_params = dict(top=0.94, bottom=0.08, left=0.06, right=0.95, hspace=0.93, wspace=0.20, )
    figsize = (9.28, 6.01)

    fig, axes = plt.subplots(6,2, figsize=figsize, sharex=True)
    fig.subplots_adjust(**sub_params)

    for i in range(6):
        axes[i,0].plot(data_dict["measured_jp"][:,i], label="measured")
        axes[i,0].plot(cal_jp[:,i], label="IK")
        axes[i,0].set_title(f"Joint {i+1}")

        axes[i,1].plot(cal_jp[:,i]- data_dict["measured_jp"][:,i], label="IK")
        axes[i,1].set_title(f"error IK Joint {i+1}")

    axes[0,0].legend()
    plt.show() 

def calculate_ik(data_dict: Dict[str, np.ndarray]):
    measured_cp = data_dict["measured_cp"]
    all_calculated_jp = np.zeros((data_dict["measured_jp"].shape[0], 6))
    for idx in range(data_dict["measured_jp"].shape[0]):
        calculated_jp = compute_IK(convert_mat_to_frame(measured_cp[:,:,idx]))
        calculated_jp = np.array(calculated_jp)
        # TODO: remove this hack when compute_IK is fixed.
        all_calculated_jp[idx,:] = calculated_jp - np.array([0,0,0.0073,0,0,0])

    return all_calculated_jp

def demo(data_dict: Dict[str, np.ndarray]):
    measured_jp = data_dict["measured_jp"]
    measured_cp = data_dict["measured_cp"]

    # Experiment 1
    idx = 250
    calculated_cp2 = compute_FK(measured_jp[idx], 7)

    log.info("Forward kinematics")
    log.info(f"calculated_cp2 - ambf\n{calculated_cp2}")
    log.info(f"measured_cp\n{measured_cp[:,:,idx]}")
    log.info(f"Error between src FK and measured_cp\n{calculated_cp2 - measured_cp[:,:,idx]}")

    calculated_jp2 = compute_IK(convert_mat_to_frame(measured_cp[:,:,idx]))
    calculated_jp2 = np.array(calculated_jp2)

    log.info("Inverse kinematics")
    log.info(f"calculated_jp2 - ambf\n{calculated_jp2}")
    log.info(f"measured_jp\n{measured_jp[idx]}")

    log.info(f"Error between ambf and measured jp\n{calculated_jp2 - measured_jp[idx]}")

    # Experiment 2
    log.info(f"Final test")
    calculated_cp2 = compute_FK(measured_jp[idx], 7)
    calculated_jp2 = compute_IK(convert_mat_to_frame(calculated_cp2))
    log.info(calculated_jp2 - measured_jp[idx])
    log.info(calculated_jp2 - measured_jp[idx]- np.array([0,0,0.0073,0,0,0]))

def ik_example():
    record_dict = DataRecorder.create_records()
    file_path = "./data/experiments/repetabability_experiment_rosbag01/01-11-2023-20-24-30/record_001.csv"
    # file_path = "./data/experiments/repetabability_experiment_rosbag01/01-11-2023-20-28-58/record_001.csv"
    # file_path = "./data/experiments/repetabability_experiment_rosbag01/01-11-2023-20-33-24/record_001.csv"

    file_path = Path(file_path)
    assert file_path.exists(), "File does not exist"
    log.info(f"Analyzing experiment {file_path.parent.name}")

    data_dict = DataReaderFromCSV(file_path, record_dict).data_dict

    demo(data_dict)
    plot_joints(data_dict)


if __name__ == "__main__":
    ik_example()