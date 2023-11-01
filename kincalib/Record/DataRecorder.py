from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict
from tf_conversions import posemath as pm
from dvrk import psm
from kincalib.Sensors.FusionTrack import FusionTrackAbstract
from kincalib.Record.Record import (
    CartesianRecord,
    JointRecord,
    RecordCollection,
    RecordCollectionCsvSaver,
)


@dataclass
class DataRecorder:
    marker_name: str
    robot_handle: psm
    ftk_handle: FusionTrackAbstract
    data_saver: RecordCollectionCsvSaver
    save_every: int = 60

    def __post_init__(self):
        self.index = None

        self.func_dict: Dict[str, Callable] = {}
        # Optical tracker
        self.marker_measured_cp = CartesianRecord("marker_measured_cp", "marker_")
        self.func_dict[self.marker_measured_cp.record_name] = lambda: self.ftk_handle.get_data(
            self.marker_name
        )

        # Robot
        self.measured_jp = JointRecord("measured_jp", "measured_")
        self.func_dict[self.measured_jp.record_name] = self.robot_handle.measured_jp
        self.measured_cp = CartesianRecord("measured_cp", "measured_")
        self.func_dict[self.measured_cp.record_name] = self.get_measured_cp_data

        self.setpoint_jp = JointRecord("setpoint_jp", "setpoint_")
        self.func_dict[self.setpoint_jp.record_name] = self.robot_handle.setpoint_jp
        self.setpoint_cp = CartesianRecord("setpoint_cp", "setpoint_")
        self.func_dict[self.setpoint_cp.record_name] = self.get_setpoint_cp_data
        self.record_list = [
            self.marker_measured_cp,
            self.measured_jp,
            self.measured_cp,
            self.setpoint_jp,
            self.setpoint_cp,
        ]
        self.rec_collection = RecordCollection(self.record_list, data_saver=self.data_saver)

    def get_ftk_data(self):
        return self.ftk_handle.get_data(self.marker_name)

    def get_measured_cp_data(self):
        return pm.toMatrix(self.robot_handle.measured_cp())

    def get_setpoint_cp_data(self):
        return pm.toMatrix(self.robot_handle.setpoint_cp())

    def collect_data(self, index):
        self.index = index + 1
        for rec_name, action in self.func_dict.items():
            self.rec_collection.get_record(rec_name).add_data(index, action())

        if self.index % self.save_every == 0:
            self.rec_collection.save_and_clear()
