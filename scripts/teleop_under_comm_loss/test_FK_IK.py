from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from kincalib.Record.Record import Record
from kincalib.Calibration.HandEyeCalibration import HandEyeBatchProcessing
from kincalib.Record.Record import CartesianRecord, JointRecord
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.Logger import Logger
from kincalib.Motion.DvrkKin import DvrkPsmKin
from surgical_robotics_challenge.kinematics.psmIK import (
    compute_FK,
    compute_IK,
    convert_mat_to_frame,
)
import surgical_robotics_challenge
import json
import numpy as np
import kincalib

log = Logger(__name__).log


@dataclass
class DataParser:
    file_path: str
    record_dict: Dict[str, Record]

    def __post_init__(self):
        self.df = pd.read_csv(self.file_path)
        self.process_dataframe()

    def process_dataframe(self):
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        self.load_records()

    def load_records(self):
        self.data_dict: dict[str, np.ndarray] = {}
        self.data_dict["traj_index"] = self.df.loc[:, "traj_index"].to_numpy()

        for record in self.record_dict.values():
            self.extract_data(record)

    def extract_data(self, record):
        if type(record) == CartesianRecord:
            self.data_dict[record.record_name] = self.process_cartesian_csv_data(
                self.df.loc[:, record.headers].to_numpy()
            )
        elif type(record) == JointRecord:
            self.data_dict[record.record_name] = self.df.loc[
                :, record.headers
            ].to_numpy()
        else:
            raise Exception(f"{type(record)} type not supported")

    def process_cartesian_csv_data(self, data: np.ndarray):
        pose_arr = np.zeros((4, 4, data.shape[0]))
        for i in range(data.shape[0]):
            pose_arr[:4, :4, i] = np.identity(4)
            pose_arr[:3, 3, i] = data[i, :3]

            rot = Rotation3D.from_rotvec(data[i, 3:])
            pose_arr[:3, :3, i] = rot.R

        return pose_arr


def plot_joints(data_dict: Dict[str, np.ndarray]):
    cal_jp = kincalib.calculate_ik(data_dict["measured_cp"])

    sub_params = dict(
        top=0.94,
        bottom=0.08,
        left=0.06,
        right=0.95,
        hspace=0.93,
        wspace=0.20,
    )
    figsize = (9.28, 6.01)

    fig, axes = plt.subplots(6, 2, figsize=figsize, sharex=True)
    fig.subplots_adjust(**sub_params)

    for i in range(6):
        axes[i, 0].plot(data_dict["measured_js"][:, i], label="measured_js")
        axes[i, 0].plot(cal_jp[:, i], label="IK(measured_cp)")
        axes[i, 0].set_title(f"Joint {i+1}")

        axes[i, 1].plot(cal_jp[:, i] - data_dict["measured_js"][:, i], label="IK")
        axes[i, 1].set_title(f"measured_js - IK(measured_cp) for joint {i+1}")

    axes[0, 0].legend()
    plt.show()


def demo(data_dict: Dict[str, np.ndarray]):
    measured_jp = data_dict["measured_js"]
    measured_cp = data_dict["measured_cp"]

    # Experiment 1
    idx = 0
    calculated_cp2 = compute_FK(measured_jp[idx], 7)

    log.info(f"Forward kinematics at index {idx}")
    log.info(f"measured_cp from SRC FK\n{calculated_cp2}")
    log.info(f"measured_cp from dVRK\n{measured_cp[:,:,idx]}")
    log.info(
        f"Error between SRC FK and dVRK measured_cp\n{calculated_cp2 - measured_cp[:,:,idx]}"
    )
    log.info(f"\n\n")

    calculated_jp2 = compute_IK(convert_mat_to_frame(measured_cp[:, :, idx]))
    calculated_jp2 = np.array(calculated_jp2)

    log.info(f"Inverse kinematics at idx {idx}")
    log.info(f"measured_jp from SRC IK\n{calculated_jp2}")
    log.info(f"measured_jp from dVRK\n{measured_jp[idx]}")
    log.info(
        f"Error between SRC_IK and measured jp\n{calculated_jp2 - measured_jp[idx]}"
    )
    log.info(f"\n\n")

    # Experiment 2
    log.info(f"Test that FK and IK are inverse functions of each other")
    calculated_cp2 = compute_FK(measured_jp[idx], 7)
    calculated_jp2 = compute_IK(convert_mat_to_frame(calculated_cp2))
    log.info(calculated_jp2 - measured_jp[idx])
    log.info(f"\n\n")


def ik_example():
    jp_rec = JointRecord("measured_js", "measured_")
    cp_rec = CartesianRecord("measured_cp", "measured_")
    record_dict = dict(measured_js=jp_rec, measured_cp=cp_rec)

    # file_path = "/home/juan1995/temp/dvrk_data_fk/classic_tool_fixed.csv"
    file_path = "/home/juan1995/temp/dvrk_data_fk/Si_tool_fixed.csv"

    file_path = Path(file_path)
    assert file_path.exists(), "File does not exist"

    log.info(f"Analyzing experiment {file_path.parent.name}")
    data_dict = DataParser(file_path, record_dict).data_dict

    demo(data_dict)
    plot_joints(data_dict)


if __name__ == "__main__":
    ik_example()
