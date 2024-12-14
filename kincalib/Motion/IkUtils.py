from kincalib.utils.Logger import Logger
from kincalib.Kinematics import DvrkPsmKin_SRC, convert_mat_to_frame
import numpy as np

log = Logger(__name__).log

def batch_calculate_ik(
    measured_cp: np.ndarray, kinematic_model: DvrkPsmKin_SRC
) -> np.ndarray:
    """Calculate IK for array of poses

    Parameters
    ----------
    measured_cp : np.ndarray
        Array of size (4,4,N) where N is the number of poses
    """

    all_calculated_jp = np.zeros((measured_cp.shape[2], 6))
    for idx in range(measured_cp.shape[2]):
        calculated_jp = kinematic_model.compute_IK(
            convert_mat_to_frame(measured_cp[:, :, idx])
        )
        calculated_jp = np.array(calculated_jp)
        all_calculated_jp[idx, :] = calculated_jp

    return all_calculated_jp


def batch_calculate_fk(jp: np.ndarray, kin_model: DvrkPsmKin_SRC) -> np.ndarray:
    all_calculated_cp = np.zeros((4, 4, jp.shape[0]))
    for idx in range(jp.shape[0]):
        calculated_cp = kin_model.compute_FK(jp[idx, :], 7)
        calculated_cp = np.array(calculated_cp)
        all_calculated_cp[:, :, idx] = calculated_cp
    return all_calculated_cp
