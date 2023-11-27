from pathlib import Path
import click
from omegaconf import OmegaConf
from train_test_simple_net import (
    load_data,
    create_model,
    load_best_weights,
    calculate_test_metrics,
)

from train_test_simple_net_confs.structured_confs import ExperimentConfig


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
    exp_metrics = calculate_test_metrics(cfg, model, dataset_container)
    exp_metrics.to_table().print()


if __name__ == "__main__":
    test_network()
