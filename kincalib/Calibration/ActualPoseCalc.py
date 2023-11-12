from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import kincalib
from kincalib.Record.DataRecorder import DataReaderFromCSV, DataRecorder
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
class RobotActualPoseCalulator:
    index_array: np.ndarray
    measured_jp: np.ndarray
    T_RG: np.ndarray
    T_RG_actual: np.ndarray

    def __post_init__(self):
        self.calculate_error_metrics()
        self.calculate_actual_jp()

    def calculate_error_metrics(self):
        self.position_error = calculate_position_error(
            T_RG=self.T_RG, T_RG_actual=self.T_RG_actual
        )
        self.orientation_error = calculate_orientation_error(
            T_RG=self.T_RG, T_RG_actual=self.T_RG_actual
        )

    def calculate_actual_jp(self):
        self.actual_jp = kincalib.calculate_ik(self.T_RG_actual)

    def convert_to_dataframe(self) -> pd.DataFrame:
        temp_dict = dict(
            q4=self.measured_jp[:, 3],
            q5=self.measured_jp[:, 4],
            q6=self.measured_jp[:, 5],
            pos_error=self.position_error,
            orientation_error=self.orientation_error,
        )

        error_data = pd.DataFrame(temp_dict)
        return error_data

    def save_to_record(
        self,
        output_path: Path,
        pos_error_threshold: float = 8.0,
        orientation_error_threshold: float = 10.0,
    ):
        """Filter and save

        Parameters
        ----------
        output_path : Path
        pos_error_threshold : float, optional
            max pos error in mm, by default 8.0
        orientation_error_threshold : float, optional
            max orientation error in deg, by default 10.0
        """

        def below_thresholds(error_metric: records.PoseErrorMetric) -> bool:
            return (
                error_metric.position_error < pos_error_threshold
                and error_metric.orientation_error < orientation_error_threshold
            )

        records_list: List[records.Record] = []

        measured_jp_rec = records.JointRecord("measured_jp", "measured_")
        actual_jp_rec = records.JointRecord("actual_jp", "actual_")
        measured_cp_rec = records.CartesianRecord("measured_cp", "measured_")
        actual_cp_rec = records.CartesianRecord("actual_cp", "actual_")
        error_rec = records.ErrorRecord("pose_error", "error_")
        records_list += [
            measured_jp_rec,
            actual_jp_rec,
            measured_cp_rec,
            actual_cp_rec,
            error_rec,
        ]

        for i in range(self.index_array.shape[0]):
            error_metric = records.PoseErrorMetric(
                self.position_error[i], self.orientation_error[i]
            )
            if below_thresholds(error_metric):
                measured_jp_rec.add_data(self.index_array[i], self.measured_jp[i])
                actual_jp_rec.add_data(self.index_array[i], self.actual_jp[i])
                measured_cp_rec.add_data(self.index_array[i], self.T_RG[:, :, i])
                actual_cp_rec.add_data(self.index_array[i], self.T_RG_actual[:, :, i])
                error_rec.add_data(self.index_array[i], error_metric)

        saver = records.RecordCollectionCsvSaver(output_path)
        saver.save(records_list, file_name="filtered_data.csv")

    @classmethod
    def load_from_file(
        cls: RobotActualPoseCalulator, file_path: Path, hand_eye_file: Path
    ) -> RobotActualPoseCalulator:
        record_dict = DataRecorder.create_records()
        data_dict = DataReaderFromCSV(file_path, record_dict).data_dict

        traj_index = data_dict["traj_index"]
        T_RG = data_dict["measured_cp"]
        T_TM = data_dict["marker_measured_cp"]
        measured_jp = data_dict["measured_jp"]

        with open(hand_eye_file, "r") as file:
            hand_eye_data = json.load(file)

        T_GM = hand_eye_data["T_GM"]  # T_GM -> marker to gripper transform
        T_MG = np.linalg.inv(T_GM)
        T_RT = hand_eye_data["T_RT"]  # T_RT -> tracker to robot transform

        T_RG_actual = cls.calculate_robot_actual_cp(T_RT=T_RT, T_TM=T_TM, T_MG=T_MG)

        return cls(
            index_array=traj_index,
            measured_jp=measured_jp,
            T_RG=T_RG,
            T_RG_actual=T_RG_actual,
        )

    @classmethod
    def calculate_robot_actual_cp(
        cls: RobotActualPoseCalulator,
        T_RT: np.ndarray,
        T_TM: np.ndarray,
        T_MG: np.ndarray,
    ):
        T_RG_actual = np.zeros_like(T_TM)

        for idx in range(T_TM.shape[2]):
            T_RG_actual[:, :, idx] = np.identity(4)
            T_RG_actual[:, :, idx] = T_RT @ T_TM[:, :, idx] @ T_MG
            # normalize rotation matrix
            T_RG_actual[:3, :3, idx] = Rotation3D.trnorm(T_RG_actual[:3, :3, idx])

        return T_RG_actual


if __name__ == "__main__":
    print("test")
