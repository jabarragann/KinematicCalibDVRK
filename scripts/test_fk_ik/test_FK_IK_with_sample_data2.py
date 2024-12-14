from pathlib import Path
from typing import Dict, List
import numpy as np
import kincalib
from kincalib.Record.DataRecorder import SensorsDataReader, RealDataRecorder
from kincalib.utils.Logger import Logger
from kincalib.Kinematics import DvrkPsmKin_SRC, convert_mat_to_frame
from kincalib.utils.Plots import *

# To style matplotlib
from kincalib.Kinematics.robotics_toolbox_kin.DvrkKin import DvrkPsmKin

log = Logger(__name__).log


def plot_joints(data_dict: Dict[str, np.ndarray], kin_model: DvrkPsmKin_SRC):
    cal_jp = kincalib.batch_calculate_ik(data_dict["measured_cp"], kin_model)
    create_joint_plot_for_IK_FK_tests(data_dict["measured_jp"], cal_jp)


def print_some_calculations(
    data_dict: Dict[str, np.ndarray], kin_model: DvrkPsmKin_SRC
):
    measured_jp = data_dict["measured_jp"]
    measured_cp = data_dict["measured_cp"]

    # Experiment 1
    idx = 250
    calculated_cp2 = kin_model.compute_FK(measured_jp[idx], 7)

    log.info("Forward kinematics")
    log.info(f"calculated_cp2 - ambf\n{calculated_cp2}")
    log.info(f"measured_cp\n{measured_cp[:,:,idx]}")
    log.info(
        f"Error between src FK and measured_cp\n{calculated_cp2 - measured_cp[:,:,idx]}"
    )

    calculated_jp2 = kin_model.compute_IK(convert_mat_to_frame(measured_cp[:, :, idx]))
    calculated_jp2 = np.array(calculated_jp2)

    log.info("Inverse kinematics")
    log.info(f"calculated_jp2 - ambf\n{calculated_jp2}")
    log.info(f"measured_jp\n{measured_jp[idx]}")

    log.info(f"Error between ambf and measured jp\n{calculated_jp2 - measured_jp[idx]}")

    # Experiment 2
    log.info(f"Final test")
    calculated_cp2 = kin_model.compute_FK(measured_jp[idx], 7)
    calculated_jp2 = kin_model.compute_IK(convert_mat_to_frame(calculated_cp2))
    log.info(calculated_jp2 - measured_jp[idx])
    log.info(calculated_jp2 - measured_jp[idx])


from pathlib import Path

current_dir = Path(__file__).resolve().parent


def ik_example():
    record_dict = RealDataRecorder.create_records()

    kin_model = DvrkPsmKin_SRC("classic")
    file_path = current_dir / "data/raw_sensor_rosbag_09_traj3.csv"

    file_path = Path(file_path)
    assert file_path.exists(), f"File ({file_path}) does not exist"
    log.info(f"Analyzing experiment {file_path.parent.name}")

    data_dict = SensorsDataReader(file_path, record_dict).data_dict

    print_some_calculations(data_dict, kin_model)
    plot_joints(data_dict, kin_model)


if __name__ == "__main__":
    ik_example()
