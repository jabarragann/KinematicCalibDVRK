# /*
# Copied and adapted from surgical robotics challenge
#
#     \author    <jbarrag3@jh.edu>
#     \author    Juan Antonio Barragan 
#     \version   1.0
# */
# //==============================================================================

import numpy as np
from . import DH, JointType, Convention, PI_2


# THIS IS THE FK FOR THE PSM MOUNTED WITH THE LARGE NEEDLE DRIVER TOOL. THIS IS THE
# SAME KINEMATIC CONFIGURATION FOUND IN THE DVRK MANUAL. NOTE, JUST LIKE A FAULT IN THE
# MTM's DH PARAMETERS IN THE MANUAL, THERE IS A FAULT IN THE PSM's DH AS WELL. BASED ON
# THE FRAME ATTACHMENT IN THE DVRK MANUAL THE CORRECT DH CAN FOUND IN THIS FILE

# ALSO, NOTICE THAT AT HOME CONFIGURATION THE TIP OF THE PSM HAS THE FOLLOWING
# ROTATION OFFSET W.R.T THE BASE. THIS IS IMPORTANT FOR IK PURPOSES.
# R_7_0 = [ 0,  1,  0 ]
#       = [ 1,  0,  0 ]
#       = [ 0,  0, -1 ]
# Basically, x_7 is along y_0, y_7 is along x_0 and z_7 is along -z_0.

# You need to provide a list of joint positions. If the list is less that the number of joint
# i.e. the robot has 6 joints, but only provide 3 joints. The FK till the 3+1 link will be provided


class PSMKinematicData:
    def __init__(self, type):
        if type=="classic":
            L_tool = 4.162 + 0.1404
        elif type=="SI":
            L_tool = 4.162 + 0.1912
        else:
            raise Exception("Invalid tool type. Choose 'classic' or 'SI'")

        self.num_links = 7

        self.L_rcc = 4.318  # From dVRK documentation x 10
        self.L_tool = L_tool # 4.162 + 0.1404 # From dVRK documentation x 10 - Classic LND tool
        # self.L_tool = 4.162 + 0.1912  # From dVRK documentation x 10 - SI LND tool
        self.L_pitch2yaw = 0.0091  # Fixed length from the palm joint to the pinch joint
        self.L_yaw2ctrlpnt = 0.0  # Fixed length from the pinch joint to the pinch tip

        # Delta between tool tip and the Remote Center of Motion
        self.L_tool2rcm_offset = self.L_rcc - self.L_tool

        # PSM DH Params
        # alpha | a | theta | d | offset | type
        self.kinematics = [
            DH(PI_2, 0, 0, 0, PI_2, JointType.REVOLUTE, Convention.MODIFIED),
            DH(-PI_2, 0, 0, 0, -PI_2, JointType.REVOLUTE, Convention.MODIFIED),
            DH(PI_2, 0, 0, 0, -self.L_rcc, JointType.PRISMATIC, Convention.MODIFIED),
            DH(0, 0, 0, self.L_tool, 0, JointType.REVOLUTE, Convention.MODIFIED),
            DH(-PI_2, 0, 0, 0, -PI_2, JointType.REVOLUTE, Convention.MODIFIED),
            DH(-PI_2, self.L_pitch2yaw, 0, 0, -PI_2, JointType.REVOLUTE, Convention.MODIFIED,), #fmt: skip
            DH(-PI_2, 0, 0, self.L_yaw2ctrlpnt, PI_2, JointType.REVOLUTE, Convention.MODIFIED,),  # fmt: skip
        ]

    def get_link_params(self, link_num):
        if link_num < 0 or link_num > self.num_links:
            # Error
            print("ERROR, ONLY ", self.num_links, " JOINT DEFINED")
            return []
        else:
            return self.kinematics[link_num]



def compute_FK(joint_pos, up_to_link, psm_kinematic_data: PSMKinematicData):
    if up_to_link > psm_kinematic_data.num_links:
        raise "ERROR! COMPUTE FK UP_TO_LINK GREATER THAN DOF"
    j = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(joint_pos)):
        j[i] = joint_pos[i]

    T_N_0 = np.identity(4)

    for i in range(up_to_link):
        link_dh = psm_kinematic_data.get_link_params(i)
        link_dh.theta = j[i]
        T_N_0 = T_N_0 * link_dh.get_trans()

    return T_N_0


# T_7_0 = compute_FK([-0.5, 0, 0.2, 0, 0, 0])
#
# print(T_7_0)
# print("\n AFTER ROUNDING \n")
# print(round_mat(T_7_0, 4, 4, 3))
# print(round_mat(T_7_0, 4, 4, 3))
