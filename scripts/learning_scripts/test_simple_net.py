from pathlib import Path
import click
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from train_test_simple_net import (
    load_data,
    create_model,
    load_best_weights,
    calculate_test_metrics,
)
from kincalib.Metrics import MetricsCalculator
from train_test_simple_net_confs.structured_confs import ExperimentConfig
from kincalib.utils.Logger import Logger

log = Logger(__name__).log


def plot_offsets(gt_offset, pred_offset):
    fig, axes = plt.subplots(6, 1, sharex=True)
    for i in range(6):
        axes[i].plot(gt_offset[:, i], label="gt")
        axes[i].plot(pred_offset[:, i], label="pred")
        axes[i].legend()

    axes[2].set_ylabel("offset (mm)")
    plt.show()


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="outputs_hydra/train_test_simple_net_20231126_201702",
)
def test_network(config_path: Path):
    cfg: ExperimentConfig  # For Duck typing
    cfg = OmegaConf.load(config_path / ".hydra/config.yaml")
    cfg_hydra = OmegaConf.load(config_path / ".hydra/hydra.yaml")
    cfg.output_path = cfg_hydra.hydra.runtime.output_dir

    # print(OmegaConf.to_yaml(cfg))

    dataset_container = load_data(cfg)
    model = create_model(cfg, dataset_container)

    model = load_best_weights(model, cfg)
    cal_metrics = calculate_test_metrics(cfg, model, dataset_container)
    exp_metrics = cal_metrics.get_metrics_container()
    exp_metrics.to_table().print()

    calc_metrics2 = MetricsCalculator(
        cal_metrics.experiment_name,
        cal_metrics.input_jp,
        cal_metrics.gt_offset,
        np.zeros_like(cal_metrics.gt_offset),
    )
    exp_metrics2 = calc_metrics2.get_metrics_container()

    log.info(f"Error with no corrections - {exp_metrics2.experiment_name}")
    exp_metrics2.to_table(with_jp=False).print(floatfmt=".5f")

    plot_offsets(cal_metrics.gt_offset, cal_metrics.pred_offset)


if __name__ == "__main__":
    test_network()
