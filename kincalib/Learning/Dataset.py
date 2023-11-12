from __future__ import annotations
from typing import List
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import Dataset
from kincalib.utils.Logger import Logger

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")


@dataclass
class JointsDataset1(Dataset):
    """
    Input: actual_jp.
    Output: measured_jp - actual_jp

    such that actual_jp + output = measured_jp

    """

    path_lists: List[Path]
    normalizer: Normalizer = None

    def __post_init__(self) -> None:
        super().__init__()
        self.X, self.Y = self.read_files(self.path_lists)
        self.X = torch.tensor(self.X.astype(np.float32))
        self.Y = torch.tensor(self.Y.astype(np.float32))

    def read_files(self, path_list: List[Path]):
        joint_names = ["q1", "q2", "q3", "q4", "q5", "q6"]
        measured_cols = ["measured_" + joint_name for joint_name in joint_names]
        actual_cols = ["actual_" + joint_name for joint_name in joint_names]
        offset_cols = ["offset_" + joint_name for joint_name in joint_names]

        X = []
        Y = []
        for p in path_list:
            assert p.exists(), f"{p} does not exist"
            df = pd.read_csv(p)
            measured_data = df.loc[:, measured_cols].to_numpy()
            actual_data = df.loc[:, actual_cols].to_numpy()
            offset_data = df.loc[:, offset_cols].to_numpy()
            X.append(measured_data)
            Y.append(offset_data)

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        return X, Y

    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx, normalize_input=True):
        if normalize_input:
            if self.normalizer is None:
                raise ValueError("normalizer cannot be None")
            x_norm = self.normalizer(self.X[idx])
            return x_norm, self.Y[idx]

        else:
            return self.X[idx], self.Y[idx]


@dataclass
class Normalizer:
    xdata: np.ndarray
    state_dict: dict = None

    def __post_init__(self):
        """Calculate mean and std of data"""
        if self.xdata is not None:
            self.mean = self.xdata.mean(axis=0)
            self.std = self.xdata.std(axis=0)
        elif self.state_dict is not None:
            self.load_state_dict(self.state_dict)
        else:
            # log.info("state_dict and xdata cannot be None at the same time")
            raise ValueError("state_dict and xdata cannot be None at the same time")

    def __call__(self, x):
        return (x - self.mean) / self.std

    def reverse(self, x):
        return (x * self.std) + self.mean

    def get_state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, scale_values_dict):
        self.mean = np.array(scale_values_dict["mean"], dtype=np.float32)
        self.std = np.array(scale_values_dict["std"], dtype=np.float32)

    def to_json(self, path: Path):
        state_dict = self.get_state_dict()
        json_str = json.dumps(state_dict, indent=2)
        with open(path, "w") as f:
            f.write(json_str)

    @classmethod
    def from_json(cls: Normalizer, path: Path) -> Normalizer:
        with open(path, "r") as f:
            state_dict = json.load(f)
        return cls(xdata=None, state_dict=state_dict)


if __name__ == "__main__":
    exp_root = []
    exp_root.append(
        "./data/experiments/data_collection1/08-11-2023-19-23-55/filtered_dataset.csv"
    )
    exp_root.append(
        "./data/experiments/data_collection1/08-11-2023-19-33-54/filtered_dataset.csv"
    )
    exp_root.append(
        "./data/experiments/data_collection1/08-11-2023-19-52-14/filtered_dataset.csv"
    )
    exp_root = [Path(p) for p in exp_root]

    train_data = JointsDataset1(path_lists=exp_root)
    normalizer = Normalizer(train_data.X)
    train_data.set_normalizer(normalizer)

    log.info(f"Training size {len(train_data)}")

    x, y = train_data[0]
    log.info(f"number of inputs {x.shape}")
    log.info(f"number of outputs {y.shape}")
    log.info(f"x shape {x.shape}")
    log.info(f"y shape {y.shape}")

    x, y = train_data.__getitem__(slice(0, None), normalize_input=False)
    log.info(f"mean and std before normalization")
    log.info(f"train x mean\n{x.mean(axis=0)}")
    log.info(f"train x std\n{x.std(axis=0)}")

    log.info(f"Y statistics")
    log.info(f"y mean {y.mean(axis=0)}")
    log.info(f"y_std {y.std(axis=0)}")
    log.info(f"y min {y.min(axis=0)}")
    log.info(f"y max {y.max(axis=0)}")
    print()

    x, y = train_data[:]
    log.info(f"mean and std after normalization")
    log.info(f"train x mean\n{x.mean(axis=0)}")
    log.info(f"train x std\n{x.std(axis=0)}")
