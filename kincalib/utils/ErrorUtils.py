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
    position_error: np.ndarray,
    orientation_error: np.ndarray,
    pos_ax: plt.Axes,
    ori_ax: plt.Axes,
):
    pos_ax.plot(position_error)
    pos_ax.set_ylabel("Position error (mm)")
    ori_ax.plot(orientation_error)
    ori_ax.set_ylabel("Orientation error (deg)")

    [a.grid() for a in [pos_ax, ori_ax]]


def create_cartesian_error_histogram(
    pos_error_setpoint_measured: np.ndarray,
    ori_error_setpoint_measured: np.ndarray,
    pos_error_measured_actual: np.ndarray,
    ori_error_measured_actual: np.ndarray,
    stat="proportion",
    bins=50,
):
    data_dict = dict(
        pos_error_sm=pos_error_setpoint_measured,
        ori_error_sm=ori_error_setpoint_measured,
        pos_error_ma=pos_error_measured_actual,
        ori_error_ma=ori_error_measured_actual,
    )
    error_data = pd.DataFrame(data_dict)

    sub_params = dict(
        top=0.88, bottom=0.11, left=0.06, right=0.96, hspace=0.27, wspace=0.29
    )
    figsize = (13.66, 5.70)
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.subplots_adjust(**sub_params)

    fig.suptitle(f"Error distribution (N={error_data.shape[0]})")

    # fmt: off

    ## setpoint vs measured
    title = axes[0, 0 ].set_title("Setpoint vs Measured")
    title.set_position([axes[0, 0].get_position().x0+1.02, axes[0, 0].get_position().y1 + 0.02])
    sns.histplot(
        data=error_data, x="pos_error_sm", ax=axes[0, 0 ], stat=stat, kde=True, bins=bins,
    )
    sns.histplot(
        data=error_data, x="ori_error_sm", ax=axes[0, 1 ], stat=stat, kde=True, bins=bins,
    )
    sns.histplot(
        data=error_data, x="pos_error_sm", ax=axes[1, 0 ], stat=stat, kde=True, cumulative=True, bins=bins
    )
    sns.histplot(
         data=error_data, x="ori_error_sm", ax=axes[1, 1 ], stat=stat, kde=True, cumulative=True, bins=bins
    )

    ## measured vs actual
    title = axes[0, 0 + 2].set_title("Measured vs actual")
    title.set_position([axes[0, 0 + 2].get_position().x0+0.62, axes[0, 0 + 2].get_position().y1 + 0.02])
    sns.histplot(
        data=error_data, x="pos_error_ma", ax=axes[0, 0 + 2], stat=stat, kde=True, bins=bins,
    )
    sns.histplot(
        data=error_data, x="ori_error_ma", ax=axes[0, 1 + 2], stat=stat, kde=True, bins=bins,
    )
    sns.histplot(
        data=error_data, x="pos_error_ma", ax=axes[1, 0 + 2], stat=stat, kde=True, cumulative=True, bins=bins
    )
    sns.histplot(
         data=error_data, x="ori_error_ma", ax=axes[1, 1 + 2], stat=stat, kde=True, cumulative=True, bins=bins
    )
    # fmt: on
    plt.show()
