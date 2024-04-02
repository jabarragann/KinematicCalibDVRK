from pathlib import Path
import time
from typing import Dict
import numpy as np
import rospy, rostopic
import message_filters
import rostopic
import click
from enum import Enum
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from kincalib.Record.Record import (
    Record,
    CartesianRecord,
    JointRecord,
    RecordCollection,
    RecordCollectionCsvSaver,
)
import tf_conversions.posemath as pm


def js_msg_2_ndarray(msg: JointState) -> np.ndarray:
    return np.array(msg.position)


def pose_msg_2_ndarray(msg: PoseStamped) -> np.ndarray:
    return pm.toMatrix(pm.fromMsg(msg.pose))


class Rostopics(Enum):
    MEASURED_CP = ("measured_cp", PoseStamped, pose_msg_2_ndarray)
    MEASURED_JP = ("measured_js", JointState, js_msg_2_ndarray)


class RosSyncClient:
    def __init__(self, slop, namespace: str, record_dict: Dict[str, Record]):
        self.record_dict = record_dict
        self.subscribers = []
        self.idx = 0

        for topic in Rostopics:
            full_topic_name = namespace + topic.value[0]
            print("Subscribing to", full_topic_name)
            self.subscribers.append(
                message_filters.Subscriber(full_topic_name, topic.value[1])
            )

        # WARNING: TimeSynchronizer did not work. Use ApproximateTimeSynchronizer instead.
        # self.time_sync = message_filters.TimeSynchronizer(self.subscribers, 10)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=slop
        )

        self.last_time = time.time()
        self.time_sync.registerCallback(self.common_cb)
        time.sleep(0.25)

    def common_cb(self, *inputs):
        if self.idx==0:
            print("Started collecting data...")

        for msg, topic in zip(inputs, Rostopics):
            topic_name = topic.value[0]
            cb = topic.value[2]
            self.record_dict[topic_name].add_data(self.idx, cb(msg))

        self.idx += 1


@click.command()
@click.option("--slop", default=0.05, type=float)
@click.option(
    "--namespace",
    required=True,
    type=str,
    help="Namespace to append to rostopic, e.g. /PSM1/",
)
@click.option("--output_path", required=True, type=click.Path(path_type=Path))
def test_sync_client(slop: float, namespace: str, output_path: Path):
    """Test approximate syncronize message filter"""

    rospy.init_node("test_ros_client")

    jp_rec = JointRecord("measured_js", "measured_")
    cp_rec = CartesianRecord("measured_cp", "measured_")
    record_dict = dict(measured_js=jp_rec, measured_cp=cp_rec)
    csv_saver = RecordCollectionCsvSaver(output_path)

    record_list = []
    for k, v in record_dict.items():
        record_list.append(v)
    rec_collection = RecordCollection(record_list, csv_saver)

    client = RosSyncClient(slop, namespace, record_dict)

    rospy.spin()

    rec_collection.save_and_clear()


if __name__ == "__main__":
    test_sync_client()
