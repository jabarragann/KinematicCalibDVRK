from kincalib.utils.Logger import Logger
import numpy as np
from surgical_robotics_challenge.kinematics.psmKinematics import PSM_TYPE_DEFAULT, TOOL_TYPE_DEFAULT
from surgical_robotics_challenge.kinematics.psmKinematics import PSMKinematicSolver
from surgical_robotics_challenge.utils.utilities import convert_mat_to_frame 

log = Logger(__name__).log

# SRC new kinematic solver
psm_solver: PSMKinematicSolver = PSMKinematicSolver(psm_type=PSM_TYPE_DEFAULT, tool_id=TOOL_TYPE_DEFAULT)

def calculate_ik(measured_cp: np.ndarray) -> np.ndarray:
    """Calculate IK for array of poses

    Parameters
    ----------
    measured_cp : np.ndarray
        Array of size (4,4,N) where N is the number of poses
    """

    all_calculated_jp = np.zeros((measured_cp.shape[2], 6))
    for idx in range(measured_cp.shape[2]):
        calculated_jp = psm_solver.compute_IK(convert_mat_to_frame(measured_cp[:, :, idx]))
        calculated_jp = np.array(calculated_jp)
        all_calculated_jp[idx, :] = calculated_jp

    return all_calculated_jp


def calculate_fk(jp: np.ndarray) -> np.ndarray:
    all_calculated_cp = np.zeros((4, 4, jp.shape[0]))
    for idx in range(jp.shape[0]):
        calculated_cp = psm_solver.compute_FK(jp[idx, :], 7)
        calculated_cp = np.array(calculated_cp)
        all_calculated_cp[:, :, idx] = calculated_cp
    return all_calculated_cp
