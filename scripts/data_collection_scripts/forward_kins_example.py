from pathlib import Path
from typing import Dict, List
from kincalib.Record.DataRecorder import DataReaderFromCSV, DataRecorder
from kincalib.Calibration.HandEyeCalibration import Batch_Processing
from kincalib.utils.Logger import Logger
from kincalib.Motion.DvrkKin import DvrkPsmKin
import json
import numpy as np

log = Logger(__name__).log

def fk_example():
    record_dict = DataRecorder.create_records()
    file_path = "./data/experiments/repetabability_experiment_rosbag01/01-11-2023-20-24-30/record_001.csv"
    # file_path = "./data/experiments/repetabability_experiment_rosbag01/01-11-2023-20-28-58/record_001.csv"
    # file_path = "./data/experiments/repetabability_experiment_rosbag01/01-11-2023-20-33-24/record_001.csv"

    file_path = Path(file_path)
    assert file_path.exists(), "File does not exist"
    log.info(f"Analyzing experiment {file_path.parent.name}")

    data_dict = DataReaderFromCSV(file_path, record_dict).data_dict

    measured_jp = data_dict["measured_jp"]
    measured_cp = data_dict["measured_cp"]
    psm_kin = DvrkPsmKin()
    calculated_cp = psm_kin.fkine(measured_jp)

    idx = 150 
    log.info("measured from robot")
    log.info(measured_cp[:,:,idx])
    log.info("calculated with python fkine model")
    log.info(calculated_cp[idx].data[0])
    log.info("error")
    log.info(measured_cp[:,:,idx] - calculated_cp[idx].data[0])


if __name__ == "__main__":
    fk_example()