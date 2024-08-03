from surgical_robotics_challenge.kinematics.psmKinematics import PSMKinematicSolver
from surgical_robotics_challenge.utils.joint_space_trajectory_generator import JointSpaceTrajectory
from surgical_robotics_challenge.utils.utilities import *
from surgical_robotics_challenge.kinematics.psmKinematics import ToolType, TOOL_TYPE_DEFAULT
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


num_joints = 7
joint_lims = np.zeros((num_joints, 2))
joint_lims[0] = [np.deg2rad(-91.96), np.deg2rad(91.96)]
joint_lims[1] = [np.deg2rad(-60), np.deg2rad(60)]
joint_lims[2] = [0.2, 2.4]
joint_lims[3] = [np.deg2rad(-175), np.deg2rad(175)]
joint_lims[4] = [np.deg2rad(-90), np.deg2rad(90)]
joint_lims[5] = [np.deg2rad(-85), np.deg2rad(85)]
joint_lims[6] = [0.0, 0.0]
js_traj = JointSpaceTrajectory(
    num_joints=7, num_traj_points=50, joint_limits=joint_lims)
num_points = js_traj.get_num_traj_points()
num_joints = 6
tool_id = TOOL_TYPE_DEFAULT
psm_solver = PSMKinematicSolver(psm_type=tool_id, tool_id=tool_id)
for i in range(num_points):
    test_q = js_traj.get_traj_at_point(i)
    T_7_0 = psm_solver.compute_FK(test_q, 7)

    computed_q = psm_solver.compute_IK(convert_mat_to_frame(T_7_0))

    test_q = round_vec(test_q)
    T_7_0 = round_mat(T_7_0, 4, 4, 3)
    errors = [0] * num_joints
    for j in range(num_joints):
        errors[j] = test_q[j] - computed_q[j]
    print(i, ': Joint Errors from IK Solver')
    error_str = ""
    for i in range(len(errors)):
        errors[i] = round(errors[i], 2)
        if errors[i] == 0.0:
            error_str = error_str + " " + bcolors.OKGREEN + \
                str(errors[i]) + bcolors.ENDC
        else:
            error_str = error_str + " " + bcolors.FAIL + \
                str(errors[i]) + bcolors.ENDC
    # print(bcolors.WARNING + "errors" + bcolors.ENDC)
    print(error_str)