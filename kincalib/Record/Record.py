from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
from kincalib.Transforms.Rotation import Rotation3D
import pandas as pd


@dataclass
class RecordCollectionCsvSaver:
    output_path: Path

    def __post_init__(self):
        self.file_counter = 1

    def save(self, records_list: List[Record]):
        headers = ["traj_index"]
        data = []
        for r in records_list:
            headers += r.headers
            data.append(np.array(r.data_array))
        data = np.concatenate(data, axis=1)
        data = np.concatenate((np.array(r.index_array).reshape(-1, 1), data), axis=1)

        df = pd.DataFrame(data, columns=headers)
        df.to_csv(self.output_path / f"record_{self.file_counter:03d}.csv", index=False)
        self.file_counter += 1


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
        if data is None:
            pose_6d = np.empty(len(self.headers)) 
            pose_6d[:] = np.nan
        else:
            rot = Rotation3D(data[:3, :3]).as_rotvec().squeeze()
            pos = data[:3, 3].squeeze()
            pose_6d = np.concatenate((pos, rot))

        self.data_array.append(np.array(pose_6d))
        self.index_array.append(idx)

        return True


if __name__ == "__main__":
    jp_rec = JointRecord("measured_jp", "measured_")
    cp_rec = CartesianRecord("measured_cp", "measured_")
    rec_collection = RecordCollection([jp_rec, cp_rec])

    print(jp_rec.headers)
    print(cp_rec.headers)
