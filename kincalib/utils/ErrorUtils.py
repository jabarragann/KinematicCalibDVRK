import numpy as np
from kincalib.Transforms.Rotation import Rotation3D

def calculate_position_error(T_RG:np.ndarray, T_RG_actual:np.ndarray):
    return np.linalg.norm((T_RG_actual - T_RG)[:3,3,:],axis=0)*1000

def calculate_orientation_error(T_RG:np.ndarray, T_RG_actual:np.ndarray):
    orientation_error = []
    for i in range(T_RG.shape[ 2 ]):
        T_RG_ith = Rotation3D(Rotation3D.trnorm(T_RG[:3,:3,i]))
        T_RG_actual_ith = Rotation3D(Rotation3D.trnorm(T_RG_actual[:3,:3, i]))
        axis_angle = (T_RG_ith.T @ T_RG_actual_ith).as_rotvec()
        orientation_error.append( np.linalg.norm(axis_angle))
    
    return np.array(orientation_error) * 180/np.pi