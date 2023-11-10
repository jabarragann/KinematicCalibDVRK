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
import kincalib.Record as records
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

log = Logger(__name__).log

@dataclass 
class ExperimentalData:
    index_array:np.ndarray
    measured_jp: np.ndarray 
    T_RG:np.ndarray 
    T_RG_actual:np.ndarray

    def __post_init__(self):
        self.calculate_error_metrics()
        self.calculate_actual_jp()

    def calculate_error_metrics(self):
        self.position_error = calculate_position_error(T_RG=self.T_RG, T_RG_actual=self.T_RG_actual)
        self.orientation_error = calculate_orientation_error(T_RG=self.T_RG, T_RG_actual=self.T_RG_actual)
    
    def calculate_actual_jp(self):
        self.actual_jp = kincalib.calculate_ik(self.T_RG_actual)
    
    def convert_to_dataframe(self)->pd.DataFrame:
        temp_dict = dict(q4 = self.measured_jp[:,3], 
        q5 = self.measured_jp[:,4],
        q6 = self.measured_jp[:,5], 
        pos_error = self.position_error,
        orientation_error = self.orientation_error)

        error_data = pd.DataFrame(temp_dict)
        return error_data
    
    def save_to_record(self, output_path:Path, pos_error_threshold:float=8.0, orientation_error_threshold:float=10.0):
        """Filter and save

        Parameters
        ----------
        output_path : Path
        pos_error_threshold : float, optional
            max pos error in mm, by default 8.0
        orientation_error_threshold : float, optional
            max orientation error in deg, by default 10.0
        """

        def below_thresholds(error_metric:records.PoseErrorMetric)->bool:
            return error_metric.position_error < pos_error_threshold \
                    and error_metric.orientation_error < orientation_error_threshold

        records_list: List[records.Record] = []

        measured_jp_rec = records.JointRecord("measured_jp", "measured_")
        actual_jp_rec = records.JointRecord("actual_jp", "actual_")
        measured_cp_rec = records.CartesianRecord("measured_cp", "measured_")
        actual_cp_rec = records.CartesianRecord("actual_cp", "actual_")
        error_rec = records.ErrorRecord("pose_error", "error_")
        records_list+=[measured_jp_rec, actual_jp_rec, measured_cp_rec, actual_cp_rec, error_rec]

        for i in range(self.index_array.shape[0]):
            error_metric = records.PoseErrorMetric(self.position_error[i], self.orientation_error[i])
            if below_thresholds(error_metric):
                measured_jp_rec.add_data(self.index_array[i], self.measured_jp[i])
                actual_jp_rec.add_data(self.index_array[i], self.actual_jp[i])
                measured_cp_rec.add_data(self.index_array[i], self.T_RG[:,:,i])
                actual_cp_rec.add_data(self.index_array[i], self.T_RG_actual[:,:,i])
                error_rec.add_data(self.index_array[i], error_metric )

        saver = records.RecordCollectionCsvSaver(output_path)
        saver.save(records_list, file_name="filtered_data.csv")
         
    
    @classmethod
    def load_from_file(cls:ExperimentalData, file_path:Path, hand_eye_file:Path)->ExperimentalData:

        record_dict = DataRecorder.create_records()
        data_dict = DataReaderFromCSV(file_path, record_dict).data_dict

        traj_index = data_dict["traj_index"]
        T_RG = data_dict["measured_cp"]
        T_TM = data_dict["marker_measured_cp"]
        measured_jp = data_dict["measured_jp"]

        with open(hand_eye_file, 'r') as file:
            hand_eye_data = json.load(file)

        T_GM = hand_eye_data["T_GM"] #T_GM -> marker to gripper transform
        T_MG = np.linalg.inv(T_GM) 
        T_RT = hand_eye_data["T_RT"] #T_RT -> tracker to robot transform

        T_RG_actual = cls.calculate_robot_actual_cp(T_RT=T_RT, T_TM=T_TM, T_MG=T_MG)

        return cls(index_array = traj_index, measured_jp=measured_jp, 
                    T_RG=T_RG, T_RG_actual=T_RG_actual)

    @classmethod
    def calculate_robot_actual_cp(cls:ExperimentalData, T_RT:np.ndarray, T_TM:np.ndarray, T_MG:np.ndarray):
        T_RG_actual = np.zeros_like(T_TM)

        for idx in range(T_TM.shape[2]):
            T_RG_actual[:,:,idx] = np.identity(4) 
            T_RG_actual[:,:,idx] = T_RT @ T_TM[:,:,idx] @ T_MG
            # normalize rotation matrix
            T_RG_actual[:3,:3,idx] = Rotation3D.trnorm(T_RG_actual[:3,:3,idx])
        
        return T_RG_actual


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

    [a.grid() for a in ax.flatten()]
    # major_ticks = np.arange(0,350,25) 
    # [a.set_xticks(major_ticks) for a in ax.flatten()]

    # correlation plot
    fig, ax = plt.subplots(3,2)
    sns.scatterplot(x="q4", y="pos_error", data=error_data, ax=ax[0,0])
    sns.scatterplot(x="q5", y="pos_error", data=error_data, ax=ax[1,0])
    sns.scatterplot(x="q6", y="pos_error", data=error_data, ax=ax[2,0])

    sns.scatterplot(x="q4", y="orientation_error", data=error_data, ax=ax[0,1])
    sns.scatterplot(x="q5", y="orientation_error", data=error_data, ax=ax[1,1])
    sns.scatterplot(x="q6", y="orientation_error", data=error_data, ax=ax[2,1])

    fig, ax = plt.subplots(2,2)
    fig.suptitle(f"Error distribution (N={error_data.shape[0]})")
    # ax = np.expand_dims(ax, axis=0)
    stat = "proportion"
    sns.histplot(data=error_data, x="pos_error", ax=ax[0,0], stat=stat, kde=True)
    sns.histplot(data=error_data, x="orientation_error", ax=ax[0,1], stat=stat, kde=True)
    sns.histplot(data=error_data, x="pos_error", ax=ax[1,0], stat=stat, kde=True, cumulative=True)
    sns.histplot(data=error_data, x="orientation_error", ax=ax[1,1], stat=stat, kde=True, cumulative=True)
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
    # exp_root = "./data/experiments/repeatability_experiment_rosbag01/01-11-2023-20-33-24"


    # exp_root = "./data/experiments/data_collection1/08-11-2023-19-23-55"
    # exp_root = "./data/experiments/data_collection1/08-11-2023-19-33-54"
    exp_root = "./data/experiments/data_collection1/08-11-2023-19-52-14"
    

    exp_root= Path(exp_root )

    file_path = exp_root / "combined_data.csv"
    hand_eye_file = exp_root / "hand_eye_calib.json"

    assert file_path.exists(), "File does not exist"
    assert hand_eye_file.exists(), "Hand eye file does not exist"

    log.info(f"Analyzing experiment {file_path.parent.name}")

    experimental_data = ExperimentalData.load_from_file(file_path=file_path, hand_eye_file=hand_eye_file)

    experimental_data.save_to_record(output_path=exp_root)
    plot_robot_error(experimental_data)
    plot_correction_offset(experimental_data)

if __name__ == "__main__":
    analyze_robot_error()