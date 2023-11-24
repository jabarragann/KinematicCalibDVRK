from __future__ import annotations
from pathlib import Path
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


@click.command()
@click.option(
    "--data_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    default="outputs_hydra/replay_motions_with_nn_20231121_220422",
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
    plot_cartesian_errors( simulated_robot_poses1, "actual_cp", simulated_robot_poses2, "actual_cp",)
    plot_cartesian_errors( real_robot_poses, "actual_cp", simulated_robot_poses2, "actual_cp",) 
    plot_cartesian_errors( real_robot_poses, "actual_cp", simulated_robot_poses2, "setpoint_cp",)
    plot_cartesian_errors( real_robot_poses, "setpoint_cp", simulated_robot_poses2, "setpoint_cp",)

    # plot_joint_errors( real_robot_poses, "actual_jp", simulated_robot_poses2, "actual_jp")
    # fmt: on

    plt.show()


if __name__ == "__main__":
    main()
