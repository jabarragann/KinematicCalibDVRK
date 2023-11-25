from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import kincalib
from kincalib.Record.DataRecorder import SensorsDataReader, RealDataRecorder
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.Logger import Logger
from kincalib.utils import calculate_orientation_error, calculate_position_error
from kincalib.Motion.IkUtils import calculate_fk
import kincalib.Record as records
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

log = Logger(__name__).log


# TODO: Improve name of attributes. T_RG -> measured_cp
@dataclass
class RobotPosesContainer:
    robot_type: str  # "real" or "sim"
    index_array: np.ndarray
    measured_jp: np.ndarray
    actual_jp: np.ndarray
    setpoint_jp: np.ndarray
    measured_cp: np.ndarray
    actual_cp: np.ndarray
    setpoint_cp: np.ndarray

    def __post_init__(self):
        self.calculate_error_metrics()

    def calculate_error_metrics(self):
        self.position_error_actual_measured = calculate_position_error(
            T_RG=self.measured_cp, T_RG_actual=self.actual_cp
        )
        self.orientation_error_actual_measured = calculate_orientation_error(
            T_RG=self.measured_cp, T_RG_actual=self.actual_cp
        )
        self.position_error_measured_setpoint = calculate_position_error(
            T_RG=self.measured_cp, T_RG_actual=self.setpoint_cp
        )
        self.orientation_error_measured_setpoint = calculate_orientation_error(
            T_RG=self.measured_cp, T_RG_actual=self.setpoint_cp
        )

    @classmethod
    def calculate_actual_jp(cls, actual_cp) -> np.ndarray:
        actual_jp = kincalib.calculate_ik(actual_cp)
        return actual_jp

    def convert_to_dataframe(self) -> pd.DataFrame:
        temp_dict = dict(
            q4=self.measured_jp[:, 3],
            q5=self.measured_jp[:, 4],
            q6=self.measured_jp[:, 5],
            pos_error=self.position_error_actual_measured,
            orientation_error=self.orientation_error_actual_measured,
        )

        error_data = pd.DataFrame(temp_dict)
        return error_data

    def convert_to_error_dataframe(self) -> pd.DataFrame:
        data_dict = dict(
            pos_error_actual_measured=self.position_error_actual_measured,
            ori_error_actual_measured=self.orientation_error_actual_measured,
            pos_error_measured_setpoint=self.position_error_measured_setpoint,
            ori_error_measured_setpoint=self.orientation_error_measured_setpoint,
            label=self.robot_type,
        )
        error_df = pd.DataFrame(data_dict)
        return error_df

    @classmethod
    def create_records_for_saving(cls) -> Dict[str, records.Record]:
        setpoint_jp = records.JointRecord("setpoint_jp", "setpoint_")
        measured_jp = records.JointRecord("measured_jp", "measured_")
        actual_jp = records.JointRecord("actual_jp", "actual_")
        setpoint_cp = records.CartesianRecord("setpoint_cp", "setpoint_")
        measured_cp = records.CartesianRecord("measured_cp", "measured_")
        actual_cp = records.CartesianRecord("actual_cp", "actual_")
        measured_setpoint_error = records.ErrorRecord(
            "measured_setpoint_error", "measured_setpoint_"
        )
        actual_measured_error = records.ErrorRecord(
            "actual_measured_error", "actual_measured_"
        )

        record_dict = dict(
            setpoint_jp=setpoint_jp,
            measured_jp=measured_jp,
            actual_jp=actual_jp,
            setpoint_cp=setpoint_cp,
            measured_cp=measured_cp,
            actual_cp=actual_cp,
            measured_setpoint_error=measured_setpoint_error,
            actual_measured_error=actual_measured_error,
        )

        assert cls.does_record_names_match_dict_keys(
            record_dict
        ), "Record names must match dict keys"

        return record_dict

    @classmethod
    def does_record_names_match_dict_keys(
        cls, record_dict: Dict[str, records.Record]
    ) -> bool:
        for name, record in record_dict.items():
            if name != record.record_name:
                return False
        return True

    def filter_and_save_to_record(
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

        records_dict = self.create_records_for_saving()

        for i in range(self.index_array.shape[0]):
            actual_measured_error = records.PoseErrorMetric(
                self.position_error_actual_measured[i],
                self.orientation_error_actual_measured[i],
            )
            measured_setpoint_error = records.PoseErrorMetric(
                self.position_error_measured_setpoint[i],
                self.orientation_error_measured_setpoint[i],
            )

            if below_thresholds(actual_measured_error):
                # fmt:off
                records_dict["setpoint_jp"].add_data( self.index_array[i], self.setpoint_jp[i])
                records_dict["measured_jp"].add_data( self.index_array[i], self.measured_jp[i])
                records_dict["actual_jp"].add_data( self.index_array[i], self.actual_jp[i])
                records_dict["setpoint_cp"].add_data( self.index_array[i], self.setpoint_cp[:, :, i])
                records_dict["measured_cp"].add_data( self.index_array[i], self.measured_cp[:, :, i])
                records_dict["actual_cp"].add_data( self.index_array[i], self.actual_cp[:, :, i])
                records_dict["measured_setpoint_error"].add_data(self.index_array[i], measured_setpoint_error)
                records_dict["actual_measured_error"].add_data(self.index_array[i], actual_measured_error)
                # fmt:on

        saver = records.RecordCollectionCsvSaver(output_path.parent)
        saver.save(list(records_dict.values()), file_name=output_path.name)

    @classmethod
    def create_from_jp_values(
        cls: RobotPosesContainer,
        robot_type: str,
        index_array: np.ndarray,
        setpoint_jp: np.ndarray,
        measured_jp: np.ndarray,
        actual_jp: np.ndarray,
    ) -> RobotPosesContainer:
        setpoint_cp = calculate_fk(setpoint_jp)
        measured_cp = calculate_fk(measured_jp)
        actual_cp = calculate_fk(actual_jp)

        cls: RobotPosesContainer
        instance = RobotPosesContainer(
            robot_type,
            index_array,
            measured_jp,
            actual_jp,
            setpoint_jp,
            measured_cp,
            actual_cp,
            setpoint_cp,
        )
        return instance

    @classmethod
    def create_from_real_measurements(
        cls: RobotPosesContainer, file_path: Path, hand_eye_file: Path
    ) -> RobotPosesContainer:
        record_dict = RealDataRecorder.create_records()
        data_dict = SensorsDataReader(file_path, record_dict).data_dict

        traj_index = data_dict["traj_index"]
        T_RG = data_dict["measured_cp"]
        T_TM = data_dict["marker_measured_cp"]
        measured_jp = data_dict["measured_jp"]
        setpoint_jp = data_dict["setpoint_jp"]
        setpoint_cp = data_dict["setpoint_cp"]

        with open(hand_eye_file, "r") as file:
            hand_eye_data = json.load(file)

        T_GM = hand_eye_data["T_GM"]  # T_GM -> marker to gripper transform
        T_MG = np.linalg.inv(T_GM)
        T_RT = hand_eye_data["T_RT"]  # T_RT -> tracker to robot transform

        T_RG_actual = cls.calculate_robot_actual_cp(T_RT=T_RT, T_TM=T_TM, T_MG=T_MG)
        actual_jp = cls.calculate_actual_jp(actual_cp=T_RG_actual)

        cls: RobotPosesContainer
        return cls(
            robot_type="real-robot",
            index_array=traj_index,
            measured_jp=measured_jp,
            actual_jp=actual_jp,
            setpoint_jp=setpoint_jp,
            measured_cp=T_RG,
            actual_cp=T_RG_actual,
            setpoint_cp=setpoint_cp,
        )

    @classmethod
    def create_from_csv_file(
        cls: RobotPosesContainer, file_path: Path, robot_type: str
    ) -> RobotPosesContainer:
        log.debug(f"Loading data from {file_path}")
        record_dict = cls.create_records_for_saving()

        # Reader does not have any method to read this type of records
        del record_dict["measured_setpoint_error"]
        del record_dict["actual_measured_error"]
        data_dict = SensorsDataReader(file_path, record_dict).data_dict

        return RobotPosesContainer(
            robot_type,
            index_array=data_dict["traj_index"],
            setpoint_jp=data_dict["setpoint_jp"],
            measured_jp=data_dict["measured_jp"],
            actual_jp=data_dict["actual_jp"],
            setpoint_cp=data_dict["setpoint_cp"],
            measured_cp=data_dict["measured_cp"],
            actual_cp=data_dict["actual_cp"],
        )

    @classmethod
    def calculate_robot_actual_cp(
        cls: RobotPosesContainer,
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
