from re import I
import sys
import time
import crtk
from kincalib.utils.Logger import Logger
import rospy
import dvrk

log = Logger(__name__).log


def create_psm_handle(name,  type:str, expected_interval=0.1):
    assert type in ["ambf","dvrk"], "type must be either ambf or dvrk"

    if type == "dvrk":
        ral = crtk.ral(name + "_crtk")
        return dvrk.arm(ral=ral, arm_name=name, expected_interval=expected_interval)
    elif type == "ambf":
        from surgical_robotics_challenge.psm_arm import PSMBlocking
        from surgical_robotics_challenge.simulation_manager import SimulationManager
        name = name.lower()
        sim_manager = SimulationManager(name+"_ambf")
        return PSMBlocking(sim_manager, name=name, add_joint_errors=False)


def home_device(arm, type:str):
    assert type in ["ambf","dvrk"], "type must be either ambf or dvrk"

    if type in "dvrk":
        log.info("-- Enabling arm")
        if not arm.enable(10):
            sys.exit("-- Failed to enable within 10 seconds")
        log.info("-- Homing arm")
        if not arm.home(10):
            sys.exit("-- Failed to home within 10 seconds")


class RobotHandler:
    """Simplified arm class to replay motion, better performance than
    dvrk.arm since we're only subscribing to topics we need.
    """

    class __jaw_device:
        """Simplified jaw class to control the jaws, will not be used without the -j option"""

        def __init__(self, jaw_namespace, ral, expected_interval, operating_state_instance):
            self.__crtk_utils = crtk.utils(self, ral, expected_interval, operating_state_instance)
            self.__crtk_utils.add_move_jp()
            self.__crtk_utils.add_servo_jp()
            self.__crtk_utils.add_measured_js()

    def __init__(self, device_namespace, expected_interval):
        # ral ==> ros abstraction layer
        self.ral: crtk.ral = crtk.ral("dvrk_psm_test")
        jaw_ral: crtk.ral = self.ral.create_child("/jaw")

        # populate this class with all the ROS topics we need
        self.crtk_utils = crtk.utils(class_instance=self, ral=self.ral, expected_interval=None)

        self.crtk_utils.add_operating_state()
        self.crtk_utils.add_servo_jp()
        self.crtk_utils.add_move_jp()
        self.crtk_utils.add_servo_cp()
        self.crtk_utils.add_move_cp()
        self.crtk_utils.add_measured_js()
        self.crtk_utils.add_measured_cp()
        self.crtk_utils.add_setpoint_js()
        self.crtk_utils.add_setpoint_cp()
        self.jaw = self.__jaw_device(
            device_namespace + "/jaw", self.ral, expected_interval, operating_state_instance=self
        )

        # create node
        if not rospy.get_node_uri():
            rospy.init_node("arm_api", anonymous=True, log_level=rospy.WARN)
            time.sleep(0.2)
        else:
            rospy.logdebug(rospy.get_caller_id() + " -> ROS already initialized")

    def jaw_jp(self):
        try:
            jaw_pose = self.jaw.measured_jp()[0]
        except RuntimeWarning as e:
            log.error("Run time warning raised when reading jaw jp")
            return -505

    def home_device(self):
        print("-- Enabling arm")
        if not self.enable(10):
            sys.exit("-- Failed to enable within 10 seconds")
        print("-- Homing arm")
        if not self.home(10):
            sys.exit("-- Failed to home within 10 seconds")
