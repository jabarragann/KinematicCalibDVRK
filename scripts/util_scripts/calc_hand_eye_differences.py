from pathlib import Path
import click
import numpy as np
import json
from kincalib.utils.Logger import Logger
from kincalib.utils.ErrorUtils import (
    calculate_position_error,
    calculate_orientation_error,
)

log = Logger(__name__).log


@click.command()
@click.option("--path1", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--path2", type=click.Path(exists=True, path_type=Path), required=True)
def calc_dif(path1, path2):
    with open(path1 / "hand_eye_calib.json") as file:
        data1 = json.load(file)

    with open(path2 / "hand_eye_calib.json") as file:
        data2 = json.load(file)

    log.info(f"Calcu difference between")
    log.info(f"{path1.name}")
    log.info(f"{path2.name}")

    for key in ["T_GM", "T_RT"]:
        pose1 = np.expand_dims(np.array(data1[key]), axis=-1)
        pose2 = np.expand_dims(np.array(data2[key]), axis=-1)

        pos_error = calculate_position_error(pose1, pose2)
        orientation_error = calculate_orientation_error(pose1, pose2)

        log.info(f"{key}")
        log.info(f"position_error:    {pos_error} mm")
        log.info(f"orientation_error: {orientation_error} deg")


if __name__ == "__main__":
    calc_dif()
