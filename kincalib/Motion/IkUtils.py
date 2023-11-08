from kincalib.utils.Logger import Logger
from surgical_robotics_challenge.kinematics.psmIK import compute_FK, compute_IK, convert_mat_to_frame
import numpy as np

log = Logger(__name__).log

def calculate_ik(measured_cp:np.ndarray)->np.ndarray:
    """ Calculate IK for array of poses

    Parameters
    ----------
    measured_cp : np.ndarray
        Array of size (4,4,N) where N is the number of poses
    """

    all_calculated_jp = np.zeros((measured_cp.shape[2], 6))
    for idx in range(measured_cp.shape[2]):
        calculated_jp = compute_IK(convert_mat_to_frame(measured_cp[:,:,idx]))
        calculated_jp = np.array(calculated_jp)
        # TODO: remove this hack when compute_IK is fixed.
        all_calculated_jp[idx,:] = calculated_jp - np.array([0,0,0.0073,0,0,0])

    return all_calculated_jp