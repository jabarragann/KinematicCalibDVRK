from pathlib import Path
from kincalib.Record.DataRecorder import SensorsDataReader, RealDataRecorder
from kincalib.utils.Logger import Logger
from kincalib.Kinematics.robotics_toolbox_kin.DvrkKin import DvrkPsmKin

log = Logger(__name__).log
current_dir = Path(__file__).resolve().parent


def fk_example():
    record_dict = RealDataRecorder.create_records()
    file_path = current_dir / "data/raw_sensor_rosbag_09_traj3.csv"

    file_path = Path(file_path)
    assert file_path.exists(), "File does not exist"
    log.info(f"Analyzing experiment {file_path.parent.name}")

    data_dict = SensorsDataReader(file_path, record_dict).data_dict

    measured_jp = data_dict["measured_jp"]
    measured_cp = data_dict["measured_cp"]
    psm_kin = DvrkPsmKin()
    calculated_cp = psm_kin.fkine(measured_jp)

    idx = 100
    log.info("measured from robot")
    log.info(measured_cp[:, :, idx])
    log.info("calculated with python fkine model")
    log.info(calculated_cp[idx].data[0])
    log.info("error")
    log.info(measured_cp[:, :, idx] - calculated_cp[idx].data[0])


if __name__ == "__main__":
    fk_example()
