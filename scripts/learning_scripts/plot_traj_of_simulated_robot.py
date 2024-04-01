from __future__ import annotations
from pathlib import Path
import numpy as np

import pandas as pd
from kincalib.utils.Logger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
from kincalib.Calibration import RobotPosesContainer
import click
from kincalib.utils import calculate_orientation_error, calculate_position_error

log = Logger(__name__).log


def plot_cartesian_errors(
    poses1: RobotPosesContainer,
    poses1_type: str,
    poses2: RobotPosesContainer,
    poses2_type: str,
):
    title = f"{poses1.robot_type}_{poses1_type} vs {poses2.robot_type}_{poses2_type}"

    posi_error = calculate_position_error(
        getattr(poses1, poses1_type), getattr(poses2, poses2_type)
    )
    ori_error = calculate_orientation_error(
        getattr(poses1, poses1_type), getattr(poses2, poses2_type)
    )

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle(title)
    axes[0].plot(posi_error)
    axes[0].set_ylabel("Position error [mm]")
    axes[1].plot(ori_error)
    axes[1].set_ylabel("Orientation error [deg]")


def plot_joint_errors(
    poses1: RobotPosesContainer,
    poses1_type: str,
    poses2: RobotPosesContainer,
    poses2_type: str,
):
    title = f"{poses1.robot_type}_{poses1_type} vs {poses2.robot_type}_{poses2_type}"

    joint_set1 = getattr(poses1, poses1_type)
    joint_set2 = getattr(poses2, poses2_type)
    joint_error = joint_set1 - joint_set2

    fig, axes = plt.subplots(6, 2, sharex=True)
    fig.suptitle(title)
    for i in range(6):
        # axes[i].plot(joint_error[:, i])
        axes[i, 0].plot(joint_set1[:, i], label=f"{poses1.robot_type}_{poses1_type}")
        axes[i, 0].plot(joint_set2[:, i], label=f"{poses2.robot_type}_{poses2_type}")
        axes[i, 0].set_title(f"Joint {i} error [deg]")

        axes[i, 1].plot(joint_error[:, i], label=f"{poses1.robot_type}_{poses1_type}")
        axes[i, 1].set_title(f"Joint {i} error [deg]")


def generate_error_hist(
    real_robot_poses: RobotPosesContainer, simulated_robot_poses: RobotPosesContainer
):
    simulated_robot_poses.robot_type = "simulated"
    simulated_robot_error_df = simulated_robot_poses.convert_to_error_dataframe()
    real_robot_error_df = real_robot_poses.convert_to_error_dataframe()
    error_data = pd.concat([simulated_robot_error_df, real_robot_error_df])

    sub_params = dict(
        top=0.92,
        bottom=0.10,
        left=0.10,
        right=0.95,
        hspace=0.25,
        wspace=0.20,
    )
    figsize = (7.42, 5.38)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(**sub_params)
    stat = "proportion"
    bins = 50
    axes[0, 0].set_title("Measured-Setpoint differences")
    axes[0, 1].set_title("Actual-Measured differences")
    # fmt: off
    # approx_actual_cp-approx_measured_cp <==> actual_cp-measured_cp
    # approx_measured_cp-setpoint_cp <==> measured_cp-setpoint_cp
    sns.histplot( data=error_data, x="pos_error_measured_setpoint", ax=axes[0, 0], stat=stat, kde=True, bins=bins, hue="label",)
    sns.histplot( data=error_data, x="ori_error_measured_setpoint", ax=axes[1, 0], stat=stat, kde=True, bins=bins, hue="label",)
    sns.histplot( data=error_data, x="pos_error_actual_measured", ax=axes[0, 1], stat=stat, kde=True, bins=bins, hue="label",)
    sns.histplot( data=error_data, x="ori_error_actual_measured", ax=axes[1, 1], stat=stat, kde=True, bins=bins, hue="label",)
    # fmt: on
    axes[0, 0].set_xlabel("Ep [mm]")
    axes[0, 1].set_xlabel("Ep [mm]")
    axes[1, 0].set_xlabel("Er [deg]")
    axes[1, 1].set_xlabel("Er [deg]")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")

    fig.savefig("results/paper_plots1/error_dist_real_simulated.png", dpi=300)
    plt.show()

    # # Get last subplot params
    # sub_params = fig.subplotpars
    # dict_str = "sub_params = dict("
    # for param in ["top", "bottom", "left", "right", "hspace", "wspace"]:
    #     dict_str = dict_str + f"{param}={getattr(sub_params, param):0.2f}, "
    # dict_str = dict_str + ")"

    # # Get figure size
    # fig_size = fig.get_size_inches()
    # fig_str = f"figsize = ({fig_size[0]:0.2f}, {fig_size[1]:0.2f})"

    # print(dict_str)
    # print(fig_str)


@click.command()
@click.option(
    "--data_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    default="outputs_hydra/replay_motions_with_nn_20231201_161711",
)
def main(data_path: Path):
    real_robot_poses_path = data_path / "real_robot_poses.csv"
    simulated_robot_poses1_path = data_path / "simulated_robot_poses1.csv"
    simulated_robot_poses2_path = data_path / "simulated_robot_poses2.csv"

    # fmt:off
    assert real_robot_poses_path.exists(), f"File {real_robot_poses_path} does not exist"
    assert simulated_robot_poses1_path.exists(), f"File {simulated_robot_poses1_path} does not exist"
    assert simulated_robot_poses2_path.exists(), f"File {simulated_robot_poses2_path} does not exist"
    # fmt:on

    real_robot_poses = RobotPosesContainer.create_from_csv_file(
        real_robot_poses_path, "real"
    )
    simulated_robot_poses1 = RobotPosesContainer.create_from_csv_file(
        simulated_robot_poses1_path, "simulated1"
    )
    simulated_robot_poses2 = RobotPosesContainer.create_from_csv_file(
        simulated_robot_poses2_path, "simulated2"
    )

    # fmt: off
    # plot_cartesian_errors( simulated_robot_poses1, "actual_cp", simulated_robot_poses2, "actual_cp",)
    # plot_cartesian_errors( real_robot_poses, "actual_cp", simulated_robot_poses2, "actual_cp",) 
    # plot_cartesian_errors( real_robot_poses, "actual_cp", simulated_robot_poses2, "setpoint_cp",)
    # plot_cartesian_errors( real_robot_poses, "setpoint_cp", simulated_robot_poses2, "setpoint_cp",)
    generate_error_hist(real_robot_poses, simulated_robot_poses2)

    # plot_joint_errors( real_robot_poses, "actual_jp", simulated_robot_poses2, "actual_jp")
    # fmt: on

    plt.show()

    # Print statistics
    posi_error = calculate_position_error(
        real_robot_poses.actual_cp, simulated_robot_poses2.setpoint_cp
    )
    ori_error = calculate_orientation_error(
        real_robot_poses.actual_cp, simulated_robot_poses2.setpoint_cp
    )
    print(f"Errors without correction")
    print("position error")
    print(f"mean: {np.mean(posi_error):0.3f}")
    print(f"std: {np.std(posi_error):0.3f}")
    print(f"Median: {np.median(posi_error):0.3f}")
    print(f"Max: {np.max(posi_error):0.3f}")

    print("orientation error")
    print(f"mean: {np.mean(ori_error):0.3f}")
    print(f"std: {np.std(ori_error):0.3f}")
    print(f"Median: {np.median(ori_error):0.3f}")
    print(f"Max: {np.max(ori_error):0.3f}")

    posi_error = calculate_position_error(
        real_robot_poses.actual_cp, simulated_robot_poses2.actual_cp
    )
    ori_error = calculate_orientation_error(
        real_robot_poses.actual_cp, simulated_robot_poses2.actual_cp
    )
    print(f"Errors with correction")
    print("position error")
    print(f"mean: {np.mean(posi_error):0.3f}")
    print(f"std: {np.std(posi_error):0.3f}")
    print(f"Median: {np.median(posi_error):0.3f}")
    print(f"Max: {np.max(posi_error):0.3f}")

    print("orientation error")
    print(f"mean: {np.mean(ori_error):0.3f}")
    print(f"std: {np.std(ori_error):0.3f}")
    print(f"Median: {np.median(ori_error):0.3f}")
    print(f"Max: {np.max(ori_error):0.3f}")


if __name__ == "__main__":
    main()
