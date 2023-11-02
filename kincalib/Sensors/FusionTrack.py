"""
Fusion Track sensor handle
"""


from re import I
import rospy
import PyKDL
import std_msgs
import tf_conversions.posemath as pm
from sensor_msgs.msg import JointState
import geometry_msgs.msg
import time
import numpy as np
from typing import Dict, List, Tuple
import scipy
from scipy.spatial import distance
from pathlib import Path
import matplotlib.pyplot as plt
from kincalib.utils.Logger import Logger
import sys
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod
import click

np.set_printoptions(precision=4, suppress=True)
log = Logger(__name__).log


@dataclass
class FusionTrackAbstract(ABC):
    marker_config_file: Path

    def __post_init__(self):
        self.marker_names = self.get_tool_names_from_json(self.marker_config_file)

    @staticmethod
    def get_tool_names_from_json(marker_config_file: Path) -> List[str]:
        with open(marker_config_file, "r") as file:
            data = json.load(file)
            tool_names = [tool["name"] for tool in data["tools"]]
            return tool_names

    @abstractmethod
    def get_data(self, marker_name: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_all_data(self) -> Dict[str, np.ndarray]:
        pass


@dataclass
class MarkerSubscriber:
    marker_name: str

    def __post_init__(self):
        self.marker_pose: np.ndarray = None
        self.rostopic = "/atracsys/" + self.marker_name + "/measured_cp"
        self.__marker_subs = rospy.Subscriber(
            self.rostopic,
            geometry_msgs.msg.PoseStamped,
            self.marker_pose_cb,
        )

    def marker_pose_cb(self, msg) -> None:
        self.marker_pose = pm.toMatrix(pm.fromMsg(msg.pose))

    def get_data(self) -> np.ndarray:
        data_to_return = self.marker_pose
        self.marker_pose = None
        return data_to_return

    def has_data(self) -> bool:
        if self.marker_pose is not None:
            return True
        else:
            return False


@dataclass
class FusionTrack(FusionTrackAbstract):
    marker_config_file: Path

    def __post_init__(self):
        super().__post_init__()

        if not rospy.get_node_uri():
            rospy.init_node("ftk_500", anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + " -> ROS already initialized")

        # Create subscribers
        self.subscribers: Dict[str, MarkerSubscriber] = {}
        for name in self.marker_names:
            self.subscribers[name] = MarkerSubscriber(name)

    def get_data(self, marker_name: str) -> np.ndarray:
        """ Warning: None data can be generated if no measurement was found
        """
        return self.subscribers[marker_name].get_data()

    def get_all_data(self) -> Dict[str, np.ndarray]:
        """ Warning: None data can be generated if no measurement was found
        """
        data = {}
        for name in self.marker_names:
            data[name] = self.subscribers[name].get_data()
        return data


class FusionTrackDummy(FusionTrackAbstract):
    marker_config_file: Path

    def __post_init__(self):
        super().__post_init__()

    def get_data(self, marker_name: str):
        return np.identity(4)

    def get_all_data(self):
        data = {}
        for name in self.marker_names:
            data[name] = np.identity(4)
        return data


@click.command("cli", context_settings={"show_default": True})
@click.option(
    "--real/--sim",
    "real_sensor",
    is_flag=True,
    default=False,
    help="real or simulated sensor",
)
@click.option(
    "--marker_config",
    default="./share/markers_config.json",
    type=click.Path(exists=True, path_type=Path),
)
def main(real_sensor, marker_config: Path):
    log = Logger("utils_log").log

    if real_sensor:
        log.info("Using real sensor")
        ftk_handler = FusionTrack(marker_config)
    else:
        log.info("Using simulated sensor")
        ftk_handler = FusionTrackDummy(marker_config)

    time.sleep(0.2)
    data_dict = ftk_handler.get_all_data()
    log.info(data_dict)


if __name__ == "__main__":
    import click

    main()
