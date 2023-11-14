import rospy
import time
import numpy as np
from kincalib.utils.Logger import Logger
from kincalib.Sensors.FusionTrack import MarkerSubscriber

np.set_printoptions(precision=4, suppress=True)
log = Logger(__name__).log


if __name__ == "__main__":

    rospy.init_node("test")
    time.sleep(0.2)

    # debug option will print message when data is received
    m = MarkerSubscriber("dvrk_tip_frame", debug=True)
    while not rospy.is_shutdown():
        rospy.sleep(0.2)

    print("exit")