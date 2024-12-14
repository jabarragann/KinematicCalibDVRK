from __future__ import annotations

# Python
from pathlib import Path
import time
from dataclasses import dataclass, field
import numpy as np
import numpy
from typing import List

# ros
from sensor_msgs.msg import JointState
from kincalib.Kinematics.robotics_toolbox_kin.DvrkKin import DvrkPsmKin

# Custom
from kincalib.Motion.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Motion.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Motion.ReplayDevice import RobotHandler
from kincalib.Motion.ReplayDevice import create_psm_handle, home_device

log = Logger(__name__).log


class TrajectoryPlayer:
    def __init__(
        self,
        replay_device: RobotHandler,
        trajectory: Trajectory,
        before_motion_loop_cb: List = [],
        after_motion_cb: List = [],
    ):
        """Class to playback a `Trajectory` using a `ReplayDevice`. This class only uses the joint
        states to replay the trajectory, no cartesian setpoints are used. An additional callback system is used
        to add extra functionalities before the motion loop and after each motion.

        See below the signature of callback functions.
        ```
        def after_motion_cb(index=None):
            pass
        ```
        Parameters
        ----------
        replay_device : RobotHandler
           Device that will be executing the trajectory
        trajectory : Trajectory
            Object that contains a collection of setpoints
        before_motion_loop_cb : List, optional
            List of callback functions that will be executed before the motion loop
        after_motion_cb : List, optional
            List of callback functions that will be executed after each motion in the motion loop

        """
        self.trajectory = trajectory
        self.replay_device = replay_device
        self.after_motion_cb = after_motion_cb
        self.before_motion_loop_cb = before_motion_loop_cb

    def replay_trajectory(self, execute_cb: bool = True):
        start_time = time.time()
        last_bag_time = self.trajectory[0].header.stamp.to_sec()

        # Before motion callbacks
        if execute_cb:
            [cb() for cb in self.before_motion_loop_cb]

        for index, new_js in enumerate(self.trajectory):
            # record start time
            loop_start_time = time.time()
            # compute expected dt
            new_bag_time = self.trajectory[index].header.stamp.to_sec()
            delta_bag_time = new_bag_time - last_bag_time
            last_bag_time = new_bag_time

            # Move
            log.info(f"Executed step {index}")
            log.info(
                f"-- Trajectory Progress --> {100*index/len(self.trajectory):0.02f} %"
            )
            self.replay_device.move_jp(
                numpy.array(new_js.position)
            ).wait()  # wait until motion is finished
            time.sleep(0.2)  # Wait to a static measurement.

            # After motion callbacks
            if execute_cb:
                [cb(**{"index": index}) for cb in self.after_motion_cb]

            # try to keep motion synchronized
            loop_end_time = time.time()
            sleep_time = delta_bag_time - (loop_end_time - loop_start_time)
            # if process takes time larger than console rate, don't sleep
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Safety sleep
            if loop_end_time - loop_start_time < 0.1:
                time.sleep(0.1)

        log.info("Time to replay trajectory: %f seconds" % (time.time() - start_time))


@dataclass
class Trajectory:
    """Class storing a collection of joint setpoints to reproduce a trajectory. This class is iterable and should be use
    with the `TrajectoryPlayer`.


    Parameters
    ----------
    sampling_factor: int
       Sample the setpoints every sampling_factor.

    """

    sampling_factor: int = 1
    bbmin: np.ndarray = np.zeros(3)
    bbmax: np.ndarray = np.zeros(3)
    last_message_time: float = 0.0
    out_of_order_counter: int = 0
    setpoints: list = field(default_factory=lambda: [])
    setpoint_js_t: str = ""
    setpoint_cp_t: str = ""

    def __post_init__(self) -> None:
        # parse bag and create list of points
        pass

    def trajectory_report(self):
        log.info("Trajectory report:")
        # report out of order setpoints
        if self.out_of_order_counter > 0:
            log.info(
                "-- Found and removed %i out of order setpoints"
                % (self.out_of_order_counter)
            )

        # convert to mm
        bbmin = self.bbmin * 1000.0
        bbmax = self.bbmax * 1000.0
        log.info(
            "-- Range of motion in mm:\n   X:[%f, %f]\n   Y:[%f, %f]\n   Z:[%f, %f]"
            % (bbmin[0], bbmax[0], bbmin[1], bbmax[1], bbmin[2], bbmax[2])
        )

        # compute duration
        duration = (
            self.setpoints[-1].header.stamp.to_sec()
            - self.setpoints[0].header.stamp.to_sec()
        )
        log.info("-- Duration of trajectory: %f seconds" % (duration))

        # Number of poses
        log.info(
            "-- Found %i setpoints using topic %s"
            % (len(self.setpoints), self.setpoint_js_t)
        )
        if len(self.setpoints) == 0:
            log.error("-- No trajectory found!")

    def __iter__(self):
        self.iteration_idx = 0
        return self

    def __next__(self):
        if self.iteration_idx * self.sampling_factor < len(self.setpoints):
            result = self.setpoints[self.iteration_idx * self.sampling_factor]
            self.iteration_idx += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, i):
        return self.setpoints[i]

    def __len__(self):
        return int(len(self.setpoints) / self.sampling_factor)

    @classmethod
    def from_ros_bag(
        cls, rosbag_handle: RosbagUtils, namespace="PSM2", sampling_factor: int = 1
    ) -> Trajectory:
        bbmin = np.zeros(3)
        bbmax = np.zeros(3)
        last_message_time = 0.0
        out_of_order_counter = 0
        setpoints = []
        setpoint_js_t = f"/{namespace}/setpoint_js"
        setpoint_cp_t = f"/{namespace}/setpoint_cp"

        # rosbag_handle = RosbagUtils(filename)
        log.info("-- Parsing bag %s" % (rosbag_handle.name))
        for bag_topic, bag_message, t in rosbag_handle.rosbag_handler.read_messages():
            # Collect setpoint_cp only to keep track of workspace
            if bag_topic == setpoint_cp_t:
                # check order of timestamps, drop if out of order
                transform_time = bag_message.header.stamp.to_sec()
                if transform_time <= last_message_time:
                    out_of_order_counter = out_of_order_counter + 1
                else:
                    # keep track of workspace
                    position = numpy.array(
                        [
                            bag_message.pose.position.x,
                            bag_message.pose.position.y,
                            bag_message.pose.position.z,
                        ]
                    )
                    if len(setpoints) == 1:
                        bbmin = position
                        bmax = position
                    else:
                        bbmin = numpy.minimum(bbmin, position)
                        bbmax = numpy.maximum(bbmax, position)
            elif bag_topic == setpoint_js_t:
                # check order of timestamps, drop if out of order
                transform_time = bag_message.header.stamp.to_sec()
                if transform_time <= last_message_time:
                    out_of_order_counter = out_of_order_counter + 1
                else:
                    setpoints.append(bag_message)

        return Trajectory(
            sampling_factor=sampling_factor,
            bbmin=bbmin,
            bbmax=bbmax,
            last_message_time=last_message_time,
            out_of_order_counter=out_of_order_counter,
            setpoints=setpoints,
            setpoint_js_t=setpoint_js_t,
            setpoint_cp_t=setpoint_cp_t,
        )

    @classmethod
    def from_setpoint_array(cls: Trajectory, setpoints_array: np.ndarray) -> Trajectory:
        setpoints_msg_list =[]
        for i in range(setpoints_array.shape[0]):
            setpoint = JointState()
            setpoint.name = ["q1", "q2", "q3", "q4", "q5", "q6"]
            setpoint.position = setpoints_array[i,:].tolist()
            setpoints_msg_list.append(setpoint)

        return Trajectory(setpoints=setpoints_msg_list)


@dataclass
class RandomJointTrajectory(Trajectory):
    class PsmJointLimits:
        # Specified in rad
        q1_range = np.array([-0.60, 0.70])
        q2_range = np.array([-0.49, 0.47])
        q3_range = np.array([0.09, 0.20])
        q4_range = np.array([-1.0, 0.3])
        q5_range = np.array([-1.2, 1.2])
        q6_range = np.array([-1.2, 1.2])

    @staticmethod
    def generate_random_joint():
        limits = RandomJointTrajectory.PsmJointLimits
        q1 = np.random.uniform(limits.q1_range[0], limits.q1_range[1])
        q2 = np.random.uniform(limits.q2_range[0], limits.q2_range[1])
        q3 = np.random.uniform(limits.q3_range[0], limits.q3_range[1])
        q4 = np.random.uniform(limits.q4_range[0], limits.q4_range[1])
        q5 = np.random.uniform(limits.q5_range[0], limits.q5_range[1])
        q6 = np.random.uniform(limits.q6_range[0], limits.q6_range[1])
        return [q1, q2, q3, q4, q5, q6]

    @classmethod
    def generate_trajectory(cls, samples: int):
        setpoints = []
        for i in range(samples):
            setpoint = JointState()
            setpoint.name = ["q1", "q2", "q3", "q4", "q5", "q6"]
            setpoint.position = RandomJointTrajectory.generate_random_joint()
            setpoints.append(setpoint)

        return RandomJointTrajectory(sampling_factor=1, setpoints=setpoints)


@dataclass
class SoftRandomJointTrajectory(RandomJointTrajectory):
    """Sample two random joints positions, then create setpoint in betweeen the start and end goal.
    The number of samples is proportional to the cartesian distance between start and end.
    """

    max_dist = 0.2632

    def __post_init__(self) -> None:
        return super().__post_init__()

    @classmethod
    def generate_trajectory(cls, samples: int, samples_per_step=18):
        setpoints = []

        init_jp = RandomJointTrajectory.generate_random_joint()
        psm_kin_model = DvrkPsmKin()
        count = 0

        while count < samples:
            new_jp = RandomJointTrajectory.generate_random_joint()

            # Calculate distance between start and end point
            joints = np.vstack(
                (np.array(init_jp).reshape(1, 6), np.array(new_jp).reshape(1, 6))
            )
            cp_positions = psm_kin_model.estimate_position_from_joints_array(joints)
            dist = np.linalg.norm(cp_positions[0, :] - cp_positions[1, :])

            # Set number of samples proportional to the distance between start and end. At least use 2 samples
            num = int((dist / SoftRandomJointTrajectory.max_dist) * samples_per_step)
            num = max(2, num)
            all_setpoints = np.linspace(init_jp, new_jp, num=num)

            for idx in range(all_setpoints.shape[0]):
                setpoint = JointState()
                setpoint.name = ["q1", "q2", "q3", "q4", "q5", "q6"]
                setpoint.position = all_setpoints[idx, :].tolist()
                setpoints.append(setpoint)
                count += 1

            init_jp = setpoint.position

        return SoftRandomJointTrajectory(
            sampling_factor=1, setpoints=setpoints[:samples]
        )


if __name__ == "__main__":
    rosbag_path = Path("data/dvrk_recorded_motions/pitch_exp_traj_01_test_cropped.bag")
    rosbag_handle = RosbagUtils(rosbag_path)
    # trajectory = Trajectory.from_ros_bag(rosbag_handle, sampling_factor=10)
    # trajectory = RandomJointTrajectory.generate_trajectory(50)
    trajectory = SoftRandomJointTrajectory.generate_trajectory(100, samples_per_step=28)

    log.info(f"Initial pt {np.array(trajectory.setpoints[0].position)}")
    log.info(f"Starting ts {trajectory.setpoints[0].header.stamp.to_sec()}")
    log.info(f"number of points {len(trajectory)}")
    trajectory.trajectory_report()

    arm_namespace = "PSM2"
    arm_type = "dvrk"
    # Robot handler has some issues with new crkt versions
    # arm = RobotHandler(device_namespace=arm_namespace, expected_interval=0.01)
    # arm.home_device()
    arm = create_psm_handle(arm_namespace, type=arm_type, expected_interval=0.01)
    home_device(arm, type=arm_type)

    # callback example
    # outer_js_calib_cb = OuterJointsCalibrationRecorder(
    #     replay_device=arm, save=False, expected_markers=4, root=Path("."), marker_name="none"
    # )
    # trajectory_player = TrajectoryPlayer(arm, trajectory, before_motion_loop_cb=[outer_js_calib_cb])

    trajectory_player = TrajectoryPlayer(arm, trajectory, before_motion_loop_cb=[])

    ans = input(
        'Press "y" to start data collection trajectory. Only replay trajectories that you know. '
    )
    if ans == "y":
        trajectory_player.replay_trajectory(execute_cb=True)
    else:
        exit(0)
