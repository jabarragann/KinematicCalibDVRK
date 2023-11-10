"""
Fusion Track sensor handle
"""


import rospy
from std_msgs.msg import Float64
import tf_conversions.posemath as pm
from sensor_msgs.msg import JointState
import message_filters
import geometry_msgs.msg
import time
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from kincalib.utils.Logger import Logger
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod
import click

np.set_printoptions(precision=4, suppress=True)
log = Logger(__name__).log


@dataclass
class MarkerPoseMeasurement:
    pose: np.ndarray
    reg_error: float

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
    def get_data(self, marker_name: str) -> MarkerPoseMeasurement:
        pass

    @abstractmethod
    def get_all_data(self) -> Dict[str, MarkerPoseMeasurement]:
        pass

    @abstractmethod
    def clear_data(self)->None:
        pass


@dataclass
class MarkerSubscriber:
    marker_name: str

    def __post_init__(self):
        self.marker_measurement: MarkerPoseMeasurement = None
        self.rostopic = "/atracsys/" + self.marker_name + "/measured_cp"
        self.reg_error_topic = "/atracsys/" + self.marker_name + "/registration_error"

        self.topics:List[Tuple] = [(self.rostopic, geometry_msgs.msg.PoseStamped), (self.reg_error_topic, Float64)]
        self.subscribers: List[message_filters.Subscriber] = []
        for topic in self.topics:
            self.subscribers.append(message_filters.Subscriber(topic[0], topic[1]))

        ts = message_filters.ApproximateTimeSynchronizer(self.subscribers, queue_size=5, slop=0.05)
        ts.registerCallback(self.data_callback)

    def data_callback(self, *inputs_msg):
        marker_pose = inputs_msg[0] 
        reg_error = inputs_msg[1].data
        self.marker_measurement = MarkerPoseMeasurement(pose=marker_pose, reg_error=reg_error)

    def msg_to_marker_pose(self, msg:geometry_msgs.msg.PoseStamped)->np.ndarray:
        return pm.toMatrix(pm.fromMsg(msg.pose))
        
    def get_data(self) -> np.ndarray:
        data_to_return = self.marker_pose
        self.marker_pose = None
        return data_to_return

    def has_data(self) -> bool:
        if self.marker_pose is not None:
            return True
        else:
            return False

    def clear_data(self):
        self.marker_measurement = None



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

    def get_data(self, marker_name: str) -> MarkerPoseMeasurement:
        """ Warning: None data can be generated if no measurement was found
        """
        return self.subscribers[marker_name].get_data()

    def get_all_data(self) -> Dict[str, MarkerPoseMeasurement]:
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

    def get_data(self, marker_name: str)->MarkerPoseMeasurement:
        return MarkerPoseMeasurement( np.identity(4), 0.0 )

    def get_all_data(self):
        data = {}
        for name in self.marker_names:
            data[name] = MarkerPoseMeasurement( np.identity(4), 0.0 )
        return data
    
    def clear_data(self):
        pass


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
