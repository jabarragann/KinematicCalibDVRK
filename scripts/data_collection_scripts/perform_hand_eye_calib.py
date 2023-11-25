from pathlib import Path
from typing import Dict, List
from kincalib.Record.DataRecorder import SensorsDataReader, RealDataRecorder
from kincalib.Calibration.HandEyeCalibration import HandEyeBatchProcessing
from kincalib.utils.Logger import Logger
import numpy as np
import click

log = Logger(__name__).log


def write_matrix(
    file, name, matrix: List[List[float]], data_count, total_data, fmt_str: str = ".10f"
):
    file.write(f'    "{name}": [\n')
    for row in matrix:
        # Write a row
        file.write("        [")
        for i, num in enumerate(row):
            formatted_num = f"{num:{fmt_str}}"
            if num >= 0:
                formatted_num = " " + formatted_num
            file.write(formatted_num)
            if i < len(row) - 1:
                file.write(", ")
        file.write("]")
        if row != matrix[-1]:
            file.write(",\n")
        else:
            file.write("\n")

    if data_count != total_data - 1:
        file.write("    ],\n")
    else:
        file.write("    ]\n")


def write_float(file, name, data, data_count, total_data, fmt_str: str = ".4f"):
    file.write(f'    "{name}": {data:{fmt_str}}')
    if data_count != total_data - 1:
        file.write(",\n")
    else:
        file.write("\n")


def manual_saving(path: Path, data_dict: Dict[str, List[List[float]]]):
    with open(path, "w") as file:
        file.write("{\n")
        for data_count, (name, data) in enumerate(data_dict.items()):
            if isinstance(data, list) and isinstance(data[0], list):
                write_matrix(file, name, data, data_count, len(data_dict))

            elif isinstance(data, float):
                write_float(file, name, data, data_count, len(data_dict))
            else:
                log.error(f"Data type not supported: {type(data)}")

        file.write("}\n")
        log.info("finish writing")


def save_hand_eye_to_json(
    path: Path, T_GM: np.ndarray, T_RT: np.ndarray, mean: float, std: float
):
    """Save to json file

    'description': [
            "T_AB means transform from frame B to frame A",
            "Coordinate frames descriptions:",
            "T -> Optical tracker"
            "G -> Gripper",
            "M -> Marker",
            "R -> Robot base",
    ],

    Parameters
    ----------
    path : Path
        output path
    T_GM : np.ndarray
        Transfrom from marker to gripper
    T_RT : np.ndarray
        Transform from Tracker to robot
    mean : float
        mean error
    std : float
        std error
    """
    assert path.is_dir(), "Path is not a directory"

    T_GM = T_GM.tolist()
    T_RT = T_RT.tolist()
    data = {
        "T_GM": T_GM,
        "T_RT": T_RT,
        "mean_error": mean[0],
        "std_error": std[0],
    }

    # with open(path/"hand_eye_calib.json", 'w') as file:
    #     json.dump(data, file, indent=4)

    manual_saving(path / "hand_eye_calib.json", data)


@click.command()
@click.option(
    "--data_path", type=click.Path(exists=True, path_type=Path), required=True
)
def perform_hand_eye(data_path):
    record_dict = RealDataRecorder.create_records()

    data_path = Path(data_path)
    assert data_path.exists(), "File does not exist"
    log.info(f"Analyzing experiment {data_path.parent.name}")

    data_dict = SensorsDataReader(data_path, record_dict).data_dict

    A_data = data_dict["measured_cp"]
    B_data = data_dict["marker_measured_cp"]
    X = None  # T_GM -> marker to gripper transform
    Y = None  # T_RT -> tracker to robot transform
    X_est, Y_est, Y_est_check, ErrorStats = HandEyeBatchProcessing.pose_estimation(
        A=A_data, B=B_data
    )

    x = 0
    log.info(X_est)
    log.info(Y_est)
    log.info(f"ErrorStats mean: {ErrorStats[0]} std: {ErrorStats[1]}")

    save_hand_eye_to_json(
        data_path.parent, T_GM=X_est, T_RT=Y_est, mean=ErrorStats[0], std=ErrorStats[1]
    )


if __name__ == "__main__":
    perform_hand_eye()
