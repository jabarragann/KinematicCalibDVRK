from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Dict, List
import numpy as np
import pandas as pd
from tf_conversions import posemath as pm
from dvrk import psm
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.Logger import Logger
from kincalib.Sensors import FusionTrackAbstract, MarkerPoseMeasurement
from kincalib.Record.Record import (
    CartesianRecord,
    JointRecord,
    MarkerCartesianRecord,
    Record,
    RecordCollection,
    RecordCollectionCsvSaver,
)

log = Logger(__file__).log


@dataclass
class DataRecorder:
    """
    TODO: needs some thought if needs to be more generic.
    """

    marker_name: str
    robot_handle: psm
    ftk_handle: FusionTrackAbstract
    data_saver: RecordCollectionCsvSaver
    save_every: int = 60

    def __post_init__(self):
        self.index = None

        self.func_dict: Dict[str, Callable] = {}
        self.records_dict: Dict[str, Record] = self.create_records()
        # Optical tracker
        # self.marker_measured_cp = CartesianRecord("marker_measured_cp", "marker_")
        self.func_dict[
            self.records_dict["marker_measured_cp"].record_name
        ] = self.get_ftk_data

        # Robot
        # self.measured_jp = JointRecord("measured_jp", "measured_")
        self.func_dict[
            self.records_dict["measured_jp"].record_name
        ] = self.robot_handle.measured_jp
        # self.measured_cp = CartesianRecord("measured_cp", "measured_")
        self.func_dict[
            self.records_dict["measured_cp"].record_name
        ] = self.get_measured_cp_data

        # self.setpoint_jp = JointRecord("setpoint_jp", "setpoint_")
        self.func_dict[
            self.records_dict["setpoint_jp"].record_name
        ] = self.robot_handle.setpoint_jp
        # self.setpoint_cp = CartesianRecord("setpoint_cp", "setpoint_")
        self.func_dict[
            self.records_dict["setpoint_cp"].record_name
        ] = self.get_setpoint_cp_data

        self.record_list = list(self.records_dict.values())
        self.rec_collection = RecordCollection(
            self.record_list, data_saver=self.data_saver
        )

    @classmethod
    def create_records(cls: DataRecorder) -> Dict[str, Record]:
        """TODO: This method should be replace with an enum

        class enum(Enum):
            marker_measured_cp = (CartesianRecord, "marker_measured_cp", "marker_")
            measured_jp = (JointRecord, "measured_jp", "measured_")
        """
        # Optical tracker
        marker_measured_cp = MarkerCartesianRecord("marker_measured_cp", "marker_")
        # Robot
        measured_jp = JointRecord("measured_jp", "measured_")
        measured_cp = CartesianRecord("measured_cp", "measured_")
        setpoint_jp = JointRecord("setpoint_jp", "setpoint_")
        setpoint_cp = CartesianRecord("setpoint_cp", "setpoint_")

        record_dict = dict(
            marker_measured_cp=marker_measured_cp,
            measured_jp=measured_jp,
            measured_cp=measured_cp,
            setpoint_jp=setpoint_jp,
            setpoint_cp=setpoint_cp,
        )
        assert cls.does_record_names_match_dict_keys(
            record_dict
        ), "Record names must match dict keys"
        return record_dict

    @staticmethod
    def does_record_names_match_dict_keys(record_dict: Dict[str, Record]) -> bool:
        for name, record in record_dict.items():
            if name != record.record_name:
                return False
        return True

    def get_ftk_data(self) -> MarkerPoseMeasurement:
        """
        Clear data and sleep before getting data to avoid sync issues.
        """
        self.ftk_handle.clear_data()
        time.sleep(0.1)
        return self.ftk_handle.get_data(self.marker_name)

    def get_measured_cp_data(self):
        return pm.toMatrix(self.robot_handle.measured_cp())

    def get_setpoint_cp_data(self):
        return pm.toMatrix(self.robot_handle.setpoint_cp())

    def collect_data(self, index):
        self.index = index + 1
        for rec_name, action_to_get_data in self.func_dict.items():
            self.rec_collection.get_record(rec_name).add_data(
                index, action_to_get_data()
            )

        if self.index % self.save_every == 0:
            log.info(f"Dumping data to csv file...")
            self.rec_collection.save_and_clear()


@dataclass
class DataReaderFromCSV:
    """Read data into a df and use record_dict headers to split the data."""
    file_path: str
    record_dict: Dict[str, Record]

    def __post_init__(self):
        self.df = pd.read_csv(self.file_path)
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

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
        elif type(record) == MarkerCartesianRecord:
            self.data_dict[record.record_name] = self.process_cartesian_csv_data(
                self.df.loc[:, record.headers[:6]].to_numpy()
            )

            self.data_dict["marker_reg_error"] = self.df.loc[
                :, record.headers[:6]
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


def test_reader():
    record_dict = DataRecorder.create_records()
    file_path = "./data/experiments/repetabability_experiment_rosbag01/01-11-2023-20-24-30/record_001.csv"
    file_path = Path(file_path)
    assert file_path.exists(), "File does not exist"
    data_dict = DataReaderFromCSV(file_path, record_dict).data_dict
    print(data_dict["measured_cp"].shape)
    print(data_dict["measured_cp"][:, :, 0])


if __name__ == "__main__":
    test_reader()
