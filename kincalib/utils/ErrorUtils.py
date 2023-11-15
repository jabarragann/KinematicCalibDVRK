import numpy as np
from kincalib.Transforms.Rotation import Rotation3D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def calculate_position_error(T_RG: np.ndarray, T_RG_actual: np.ndarray):
    return np.linalg.norm((T_RG_actual - T_RG)[:3, 3, :], axis=0) * 1000


def calculate_orientation_error(T_RG: np.ndarray, T_RG_actual: np.ndarray):
    orientation_error = []
    for i in range(T_RG.shape[2]):
        T_RG_ith = Rotation3D(Rotation3D.trnorm(T_RG[:3, :3, i]))
        T_RG_actual_ith = Rotation3D(Rotation3D.trnorm(T_RG_actual[:3, :3, i]))
        axis_angle = (T_RG_ith.T @ T_RG_actual_ith).as_rotvec()
        orientation_error.append(np.linalg.norm(axis_angle))

    return np.array(orientation_error) * 180 / np.pi


def create_cartesian_error_lineplot(
    position_error: np.ndarray, orientation_error: np.ndarray, pos_ax: plt.Axes, ori_ax: plt.Axes
):

    pos_ax.plot(position_error)
    pos_ax.set_ylabel("Position error (mm)")
    ori_ax.plot(orientation_error)
    ori_ax.set_ylabel("Orientation error (deg)")

    [a.grid() for a in [pos_ax, ori_ax]]


def create_cartesian_error_histogram(
    position_eror: np.ndarray, orientation_error: np.ndarray
):
    data_dict = dict(pos_error=position_eror, orientation_error=orientation_error)
    error_data = pd.DataFrame(data_dict)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f"Error distribution (N={error_data.shape[0]})")
    # ax = np.expand_dims(ax, axis=0)
    stat = "proportion"
    sns.histplot(data=error_data, x="pos_error", ax=ax[0, 0], stat=stat, kde=True)
    sns.histplot(
        data=error_data, x="orientation_error", ax=ax[0, 1], stat=stat, kde=True
    )
    sns.histplot(
        data=error_data,
        x="pos_error",
        ax=ax[1, 0],
        stat=stat,
        kde=True,
        cumulative=True,
    )
    sns.histplot(
        data=error_data,
        x="orientation_error",
        ax=ax[1, 1],
        stat=stat,
        kde=True,
        cumulative=True,
    )

    return fig, ax
