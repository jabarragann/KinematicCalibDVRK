from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import kincalib
from kincalib.Record.DataRecorder import DataReaderFromCSV, DataRecorder
from kincalib.Calibration.HandEyeCalibration import Batch_Processing
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.Logger import Logger
from kincalib.utils import calculate_orientation_error, calculate_position_error
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

log = Logger(__name__).log

@dataclass 
class ExperimentalData:
    measured_jp: np.ndarray 
    T_RG:np.ndarray 
    T_RG_actual:np.ndarray

    def __post_init__(self):
        self.calculate_error_metrics()
        self.actual_jp = self.calculate_actual_jp()

    def calculate_error_metrics(self):
        self.position_error = calculate_position_error(T_RG=self.T_RG, T_RG_actual=self.T_RG_actual)
        self.orientation_error = calculate_orientation_error(T_RG=self.T_RG, T_RG_actual=self.T_RG_actual)
    
    def calculate_actual_jp(self):
        return kincalib.calculate_ik(self.T_RG_actual)
    
    def convert_to_dataframe(self)->pd.DataFrame:
        temp_dict = dict(q4 = self.measured_jp[:,3], 
        q5 = self.measured_jp[:,4],
        q6 = self.measured_jp[:,5], 
        pos_error = self.position_error,
        orientation_error = self.orientation_error)

        error_data = pd.DataFrame(temp_dict)
        return error_data
    
    @classmethod
    def load_from_file(cls:ExperimentalData, file_path:Path, hand_eye_file:Path):

        record_dict = DataRecorder.create_records()
        data_dict = DataReaderFromCSV(file_path, record_dict).data_dict
        T_RG = data_dict["measured_cp"]
        T_TM = data_dict["marker_measured_cp"]
        measured_jp = data_dict["measured_jp"]

        with open(hand_eye_file, 'r') as file:
            hand_eye_data = json.load(file)

        T_GM = hand_eye_data["T_GM"] #T_GM -> marker to gripper transform
        T_MG = np.linalg.inv(T_GM)
        T_RT = hand_eye_data["T_RT"] #T_RT -> tracker to robot transform

        T_RG_actual = cls.calculate_robot_actual_cp(T_RT=T_RT, T_TM=T_TM, T_MG=T_MG)

        return cls(measured_jp=measured_jp, T_RG=T_RG, T_RG_actual=T_RG_actual)

    @classmethod
    def calculate_robot_actual_cp(cls:ExperimentalData, T_RT:np.ndarray, T_TM:np.ndarray, T_MG:np.ndarray):
        T_RG_actual = np.zeros_like(T_TM)

        for idx in range(T_TM.shape[2]):
            T_RG_actual[:,:,idx] = np.identity(4) 
            T_RG_actual[:,:,idx] = T_RT @ T_TM[:,:,idx] @ T_MG
        
        return T_RG_actual

    @classmethod
    def calculate_position_error(cls, T_RG:np.ndarray, T_RG_actual:np.ndarray):
        return np.linalg.norm((T_RG_actual - T_RG)[:3,3,:],axis=0)*1000

    @classmethod
    def calculate_orientation_error(cls, T_RG:np.ndarray, T_RG_actual:np.ndarray):
        orientation_error = []
        for i in range(T_RG.shape[ 2 ]):
            T_RG_ith = Rotation3D(Rotation3D.trnorm(T_RG[:3,:3,i]))
            T_RG_actual_ith = Rotation3D(Rotation3D.trnorm(T_RG_actual[:3,:3, i]))
            axis_angle = (T_RG_ith.T @ T_RG_actual_ith).as_rotvec()
            orientation_error.append( np.linalg.norm(axis_angle))
        
        return np.array(orientation_error) * 180/np.pi



def plot_robot_error(experimental_data:ExperimentalData):
    position_error = experimental_data.position_error
    orientation_error = experimental_data.orientation_error
    error_data = experimental_data.convert_to_dataframe()


    fig, ax = plt.subplots(2,1, sharex=True)
    ax = np.expand_dims(ax, axis=0)

    ax[0,0].plot(position_error)
    ax[0,0].set_ylabel("Position error (mm)")
    ax[0,1].plot(orientation_error)
    ax[0,1].set_ylabel("Orientation error (deg)")

    # for i in range(3):
    #     ax[i+1,0].plot(measured_jp[:,3+i])
    #     ax[i+1,1].plot(measured_jp[:,3+i])

    major_ticks = np.arange(0,350,25) 
    [a.grid() for a in ax.flatten()]
    [a.set_xticks(major_ticks) for a in ax.flatten()]

    # correlation plot
    fig, ax = plt.subplots(3,2)
    sns.scatterplot(x="q4", y="pos_error", data=error_data, ax=ax[0,0])
    sns.scatterplot(x="q5", y="pos_error", data=error_data, ax=ax[1,0])
    sns.scatterplot(x="q6", y="pos_error", data=error_data, ax=ax[2,0])

    sns.scatterplot(x="q4", y="orientation_error", data=error_data, ax=ax[0,1])
    sns.scatterplot(x="q5", y="orientation_error", data=error_data, ax=ax[1,1])
    sns.scatterplot(x="q6", y="orientation_error", data=error_data, ax=ax[2,1])

    plt.show()

def plot_correction_offset(experimental_data:ExperimentalData):  
    sub_params = dict(top=0.88, bottom=0.06, left=0.07, right=0.95, hspace=0.62, wspace=0.31, )
    figsize = (7.56, 5.99)

    fig, ax = plt.subplots(6,2, sharex=True,figsize=figsize)
    fig.subplots_adjust(**sub_params)

    correction_offset = experimental_data.measured_jp - experimental_data.actual_jp
    for i in range(6):
        ax[i,0].plot(experimental_data.measured_jp[:,i])
        ax[i,0].set_title(f"q{i+1}")
        ax[i,1].plot(correction_offset[:,i])
        ax[i,1].set_title(f"correction offset q{i+1}")

    plt.show()


def analyze_robot_error():
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-24-30"
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-28-58"
    exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-33-24"
    exp_root= Path(exp_root )

    file_path = exp_root / "record_001.csv"
    hand_eye_file = exp_root / "hand_eye_calib.json"

    assert file_path.exists(), "File does not exist"
    assert hand_eye_file.exists(), "Hand eye file does not exist"

    log.info(f"Analyzing experiment {file_path.parent.name}")

    experimental_data = ExperimentalData.load_from_file(file_path=file_path, hand_eye_file=hand_eye_file)

    plot_robot_error(experimental_data)
    plot_correction_offset(experimental_data)

if __name__ == "__main__":
    analyze_robot_error()