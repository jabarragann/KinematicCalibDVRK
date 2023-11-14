"""Convert data from the data collection script to the
format required by the ETHZ handeye calibration package.

https://github.com/ethz-asl/hand_eye_calibration

"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from kincalib.Record.DataRecorder import DataReaderFromCSV, DataRecorder
from kincalib.Calibration.HandEyeCalibration import HandEyeBatchProcessing
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.utils.Logger import Logger
import numpy as np
import click

log = Logger(__name__).log


def get_quat_from_data(data: np.ndarray):
    quat = np.zeros((4, data.shape[-1]))
    for idx in range(data.shape[-1]):
        quat[:, idx] = Rotation3D(rot=data[:3, :3, idx]).as_quaternion()

    return quat


def save_to_ethz_format(path: Path, ts: np.ndarray, data: np.ndarray, debug=False):
    headers = ["t", "x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
    position = data[:3, 3, :]
    orientation = get_quat_from_data(data)
    x = position[0, :]
    y = position[1, :]
    z = position[2, :]
    qx = orientation[0, :]
    qy = orientation[1, :]
    qz = orientation[2, :]
    qw = orientation[3, :]

    data_dict = dict(t=ts, x=x, y=y, z=z, q_x=qx, q_y=qy, q_z=qz, q_w=qw)
    df = pd.DataFrame(data_dict)

    df.to_csv(path, index=False)

    if debug:
        log.info(f"position shape {position.shape}")
        log.info(f"orientation shape {orientation.shape}")
        print(df.head())


@click.command()
@click.option(
    "--data_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    default="./data/experiments/data_collection2_test/chunk1/combined_data.csv",
)
def perform_hand_eye(data_path):
    record_dict = DataRecorder.create_records()

    data_path = Path(data_path)
    assert data_path.exists(), "File does not exist"
    log.info(f"Reformatting data in  {data_path}")

    data_dict = DataReaderFromCSV(data_path, record_dict).data_dict
    ts = data_dict["traj_index"]

    T_BH = data_dict["measured_cp"]
    T_WE = data_dict["marker_measured_cp"]

    saving_path = data_path.parent / "ethz_format"
    saving_path.mkdir(exist_ok=True)

    save_to_ethz_format(path=saving_path / "poses_B_H.csv", ts=ts, data=T_BH)
    save_to_ethz_format(path=saving_path / "poses_W_E.csv", ts=ts, data=T_WE)


if __name__ == "__main__":
    perform_hand_eye()
