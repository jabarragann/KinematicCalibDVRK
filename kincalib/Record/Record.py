from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from kincalib.Sensors import MarkerPoseMeasurement
from kincalib.Transforms.Rotation import Rotation3D
import pandas as pd
from collections import namedtuple


@dataclass
class PoseErrorMetric:
    position_error: float
    orientation_error: float


@dataclass
class RecordCollectionCsvSaver:
    output_path: Path

    def __post_init__(self):
        self.file_counter = 1

    def save(self, records_list: List[Record], file_name: str = None):
        headers, data, index_array = self.collect_data_from_records(records_list)

        data = np.concatenate(data, axis=1)
        data = np.concatenate((np.array(index_array).reshape(-1, 1), data), axis=1)

        df = pd.DataFrame(data, columns=headers)

        if file_name is None:
            saving_path = self.output_path / f"record_{self.file_counter:03d}.csv"
        else:
            saving_path = self.output_path / file_name

        df.to_csv(saving_path, index=False)
        self.file_counter += 1

    def collect_data_from_records(self, records_list: List[Record]):
        """Extra data will happen if the script is interrupted before querying all new data"""
        headers = ["traj_index"]
        data = []
        last_record = records_list[-1]

        pts_in_last_record = np.array(last_record.data_array).shape[0]
        for r in records_list:
            headers += r.headers

            arr = np.array(r.data_array)
            if arr.shape[0] != pts_in_last_record:
                arr = arr[:pts_in_last_record, :]
            data.append(arr)

        index_array = r.index_array

        return headers, data, index_array


@dataclass
class RecordCollection:
    records_list: List[Record]
    data_saver: RecordCollectionCsvSaver

    def __post_init__(self):
        assert self.are_all_records_names_unique(), "Record names must be unique"

        self.record_dict: Dict[str, Record] = {}
        for r in self.records_list:
            self.record_dict[r.record_name] = r

        assert self.are_all_headers_unique(), "Headers from all records must be unique"

    def get_record(self, record_name: str) -> Record:
        return self.record_dict[record_name]

    def are_all_records_names_unique(self) -> bool:
        names = [r.record_name for r in self.records_list]
        return self.are_elements_unique(names)

    def are_all_headers_unique(self) -> bool:
        all_headers = []
        for n, rec in self.record_dict.items():
            all_headers += rec.headers

        return self.are_elements_unique(all_headers)

    def save_and_clear(self):
        self.save_data()
        self.clear_data()

    def save_data(self):
        self.data_saver.save(self.records_list)

    def clear_data(self):
        for r in self.records_list:
            r.data_array = []
            r.index_array = []

    @staticmethod
    def are_elements_unique(input_list) -> bool:
        return len(input_list) == len(set(input_list))


class Record(ABC):
    def __init__(self, record_name: str, headers: List[str]):
        self.record_name = record_name
        self.headers: List[str] = headers
        self.data_array: List[np.ndarray] = []
        self.index_array: List[np.ndarray] = []

    @abstractmethod
    def add_data(self, idx: int, data: np.ndarray) -> bool:
        self.data_array.append(data)
        self.index_array.append(idx)
        return True


class JointRecord(Record):
    def __init__(self, record_name: str, header_prefix: str):
        headers = self.get_headers(header_prefix)
        super().__init__(record_name, headers)

    def get_headers(self, prefix):
        headers = ["q1", "q2", "q3", "q4", "q5", "q6"]
        return [prefix + h for h in headers]

    def add_data(self, idx: int, data: np.ndarray) -> bool:
        if data is None:
            return False

        self.data_array.append(np.array(data))
        self.index_array.append(idx)
        return True


class CartesianRecord(Record):
    """Record with a 3d vector for positions and a 3d vector for rotation (Rodriguez or axis-angle representation)."""

    def __init__(self, record_name: str, header_prefix: str):
        headers = self.get_headers(header_prefix)
        super().__init__(record_name, headers)

    def get_headers(self, prefix):
        headers = ["x", "y", "z", "rx", "ry", "rz"]
        return [prefix + h for h in headers]

    def add_data(self, idx: int, data: np.ndarray) -> bool:
        assert isinstance(
            data, np.ndarray
        ), f"Data must be of type np.ndarray for {self.record_name}"

        pose_6d = self.pose_matrix_2_rotvec_rep(data)
        self.data_array.append(np.array(pose_6d))
        self.index_array.append(idx)

        return True

    def pose_matrix_2_rotvec_rep(self, data: np.ndarray):
        encoded_pose = np.empty(6)
        if data is None:
            encoded_pose[:] = np.nan
        else:
            rot = Rotation3D(data[:3, :3]).as_rotvec().squeeze()
            pos = data[:3, 3].squeeze()
            encoded_pose[:] = np.concatenate((pos, rot))

        return encoded_pose


class MarkerCartesianRecord(CartesianRecord):
    def __init__(self, record_name: str, header_prefix: str):
        super().__init__(record_name, header_prefix)

    def get_headers(self, prefix):
        headers = ["x", "y", "z", "rx", "ry", "rz", "reg_error"]
        return [prefix + h for h in headers]

    def add_data(self, idx: int, data: MarkerPoseMeasurement) -> bool:
        assert (
            isinstance(data, MarkerPoseMeasurement) or data is None
        ), f"Data must be of type MarkerPoseMeasurement for {self.record_name}"

        final_data = np.zeros(7)
        if data is None:
            pose_6d = self.pose_matrix_2_rotvec_rep(None)
            reg_error = np.nan
        else:
            pose_6d = self.pose_matrix_2_rotvec_rep(data.pose)
            reg_error = data.reg_error

        final_data[:6] = pose_6d
        final_data[6] = reg_error
        self.data_array.append(np.array(final_data))
        self.index_array.append(idx)

        return True


class ErrorRecord(Record):
    def __init__(self, record_name: str, header_prefix: str):
        headers = self.get_headers(header_prefix)
        super().__init__(record_name, headers)

    def get_headers(self, prefix):
        headers = ["pos_error", "rot_error"]
        return [prefix + h for h in headers]

    def add_data(self, idx: int, data: PoseErrorMetric) -> bool:
        """
        Error data is a tuple of (position_error, orientation_error)
        """
        data_to_save = np.empty(2)
        if data is None:
            data_to_save[:] = np.nan
        else:
            data_to_save[:] = [data.position_error, data.orientation_error]

        self.data_array.append(np.array(data_to_save))
        self.index_array.append(idx)

        return True


if __name__ == "__main__":
    jp_rec = JointRecord("measured_jp", "measured_")
    cp_rec = CartesianRecord("measured_cp", "measured_")
    rec_collection = RecordCollection([jp_rec, cp_rec], None)

    print(jp_rec.headers)
    print(cp_rec.headers)
