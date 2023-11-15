from __future__ import annotations
from pathlib import Path
from kincalib.utils.Logger import Logger
from kincalib.utils import (
    create_cartesian_error_histogram,
    create_cartesian_error_lineplot,
)
import matplotlib.pyplot as plt
import seaborn as sns
from kincalib.Calibration import RobotActualPoseCalulator
import click

log = Logger(__name__).log


def plot_robot_error(experimental_data: RobotActualPoseCalulator):
    position_error = experimental_data.position_error_measured_actual
    orientation_error = experimental_data.orientation_error_measured_actual
    error_data = experimental_data.convert_to_dataframe()

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].set_title("Error between measured and actual")
    create_cartesian_error_lineplot(
        experimental_data.position_error_measured_actual,
        experimental_data.orientation_error_measured_actual,
        axes[0, 0],
        axes[1, 0],
    )
    axes[0, 1].set_title("Error between measured and setpoint")
    create_cartesian_error_lineplot(
        experimental_data.position_error_measured_setpoint,
        experimental_data.orientation_error_measured_setpoint,
        axes[0, 1],
        axes[1, 1],
    )

    # correlation plot
    fig, ax = plt.subplots(3, 2)
    sns.scatterplot(x="q4", y="pos_error", data=error_data, ax=ax[0, 0])
    sns.scatterplot(x="q5", y="pos_error", data=error_data, ax=ax[1, 0])
    sns.scatterplot(x="q6", y="pos_error", data=error_data, ax=ax[2, 0])

    sns.scatterplot(x="q4", y="orientation_error", data=error_data, ax=ax[0, 1])
    sns.scatterplot(x="q5", y="orientation_error", data=error_data, ax=ax[1, 1])
    sns.scatterplot(x="q6", y="orientation_error", data=error_data, ax=ax[2, 1])

    create_cartesian_error_histogram(position_error, orientation_error, bins=30)

    plt.show()


def plot_correction_offset(experimental_data: RobotActualPoseCalulator):
    sub_params = dict(
        top=0.88,
        bottom=0.06,
        left=0.07,
        right=0.95,
        hspace=0.62,
        wspace=0.31,
    )
    figsize = (7.56, 5.99)

    fig, ax = plt.subplots(6, 2, sharex=True, figsize=figsize)
    fig.subplots_adjust(**sub_params)

    correction_offset = experimental_data.measured_jp - experimental_data.actual_jp
    for i in range(6):
        ax[i, 0].plot(experimental_data.measured_jp[:, i])
        ax[i, 0].set_title(f"q{i+1}")
        ax[i, 1].plot(correction_offset[:, i])
        ax[i, 1].set_title(f"correction offset q{i+1}")

    plt.show()


@click.command()
@click.option(
    "--data_file", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--handeye_file", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--out_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="if it is not provided use data_file parent dir",
)
@click.option("--out_name", type=str, default="filtered_dataset.csv")
def analyze_robot_error(data_file, handeye_file, out_dir, out_name):
    log.info(f"Analyzing experiment {data_file.parent.name}")

    experimental_data = RobotActualPoseCalulator.load_from_file(
        file_path=data_file, hand_eye_file=handeye_file
    )
    if out_dir is None:
        experimental_data.filter_and_save_to_record(
            output_path=data_file.parent / out_name
        )
    else:
        experimental_data.filter_and_save_to_record(output_path=out_dir / out_name)

    plot_robot_error(experimental_data)
    plot_correction_offset(experimental_data)


if __name__ == "__main__":
    analyze_robot_error()
