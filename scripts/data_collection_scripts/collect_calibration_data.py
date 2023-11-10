import click
from datetime import datetime
import json
from pathlib import Path

# kincalib module
from kincalib.utils.Logger import Logger
from kincalib.Record.DataRecorder import DataRecorder
from kincalib.Record.Record import RecordCollectionCsvSaver
from kincalib.Sensors.FusionTrack import FusionTrack, FusionTrackDummy

# from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.Motion.RosbagUtils import RosbagUtils
from kincalib.Motion.ReplayDevice import create_psm_handle, home_device
from kincalib.Motion.TrajectoryPlayer import (
    SoftRandomJointTrajectory,
    TrajectoryPlayer,
    Trajectory,
    RandomJointTrajectory,
)

log = Logger("collection").log


def save_configurations(config_dict: dict, filename: Path):
    with open(filename, "w") as file:
        json.dump(config_dict, file, indent=4)
    log.info(f"Saved configurations to {filename}")


def report_and_confirm(config_dict) -> str:
    log.info("Collection information")
    for key, value in config_dict.items():
        log.info(f"{key}: {value}")
    # log.info(f"Use real_setup:    {config_dict['use_real_setup']}")
    # log.info(f"Data root dir:     {config_dict['output_path']}")
    # log.info(f"Trajectory type    {config_dict['traj_type']}")
    # log.info(f"Trajectory length: {config_dict['traj_size']}")
    # log.info(f"Rosbag:            {config_dict['rosbag_path']}")
    # log.info(f"Description:       {config_dict['description']}")

    ans = input(
        'Press "y" to start data collection trajectory. Only replay trajectories that you know. '
    )
    return ans


@click.command("main", context_settings={"show_default": True})
@click.option(
    "--marker_config",
    type=click.Path(exists=True, path_type=Path),
    default="share/markers_config.json",
    help="Marker config file",
)
@click.option("--marker_name", type=str, default="dvrk_tip_frame")
@click.option("--traj_type", type=click.Choice(["rosbag", "random", "soft"]), default="rosbag")
@click.option("--traj_size", type=int, default=50)
@click.option("--rosbag_path", type=click.Path(exists=True, path_type=Path), default="./data/dvrk_recorded_motions/pitch_exp_traj_01_test_cropped.bag")
@click.option("--description", type=str, default="")
@click.option("--arm_type", type=click.Choice(["ambf","dvrk"]), default="dvrk")
@click.option(
    "--real/--sim",
    "use_real_sensor",
    is_flag=True,
    default=False,
    help="--real to use real atracsys sensor, --sim to use dummy sensor",
)
@click.option("--save_every", type=int, default=400, help="Send data to csv every n samples")
def main(
    marker_config, marker_name, traj_type, traj_size, rosbag_path, description, use_real_sensor, arm_type, save_every
):
    # Save config
    now = datetime.now()
    output_path = Path(f"outputs/{now.strftime('%d-%m-%Y-%H-%M-%S')}/")
    output_path.mkdir(parents=True, exist_ok=True)
    config_dict = dict(
        marker_config=str(marker_config),
        marker_name=marker_name,
        traj_type=traj_type,
        traj_size=traj_size,
        rosbag_path=str(rosbag_path),
        description=description,
        use_real_sensor=use_real_sensor,
        arm_type = arm_type,
        save_every = save_every,
    )
    config_dict["output_path"] = str(output_path)

    # Create devices
    arm = create_psm_handle("PSM2", type=arm_type, expected_interval=0.01)
    home_device(arm, type=arm_type)
    if use_real_sensor:
        fusion_track = FusionTrack(marker_config)
    else:
        fusion_track = FusionTrackDummy(marker_config)

    # Load trajectory
    if traj_type == "rosbag":
        rosbag_handle = RosbagUtils(rosbag_path)
        trajectory = Trajectory.from_ros_bag(rosbag_handle, sampling_factor=10)
        config_dict['traj_size'] = len(trajectory)
    elif traj_type == "random":
        trajectory = RandomJointTrajectory.generate_trajectory(traj_size)
    elif traj_type == "soft":
        trajectory = SoftRandomJointTrajectory.generate_trajectory(traj_size)

    # Create trajectory player and recorders
    csv_saver = RecordCollectionCsvSaver(output_path)
    data_recorder = DataRecorder(marker_name, arm, fusion_track, data_saver=csv_saver, save_every=save_every)

    trajectory_player = TrajectoryPlayer(
        replay_device=arm,
        trajectory=trajectory,
        before_motion_loop_cb=[],
        after_motion_cb=[data_recorder.collect_data],
    )

    ans = report_and_confirm(config_dict)
    save_configurations(config_dict, output_path / "config.json")
    if ans == "y":
        try:
            trajectory_player.replay_trajectory(execute_cb=True)
        finally:
            data_recorder.rec_collection.save_data()


if __name__ == "__main__":
    main()
