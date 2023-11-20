from dataclasses import dataclass
from pathlib import Path
from hydra.core.config_store import ConfigStore


def register_struct_configs():
    """Enable hydra config validation using struct configs.

    Followed the example: https://hydra.cc/docs/tutorials/structured_config/schema/
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=AppConfig)


@dataclass
class AppConfig:
    workspace: Path
    output_path: Path
    global_device: str
    traj_type: str
    rosbag_path: Path
    nn_measured_setpoint_path: Path
    nn_actual_measured_path: Path
    test_data_path: Path
    hand_eye_path: Path
