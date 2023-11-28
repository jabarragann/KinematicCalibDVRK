from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from kincalib.Learning import Trainer, BestMLP2
from kincalib.utils.Logger import Logger
from omegaconf import OmegaConf
import hydra
from kincalib.Learning.Dataset import JointsDataset1, Normalizer
from torch.utils.data import DataLoader
from kincalib.Metrics import ExperimentMetrics, MetricsCalculator

# Hack to add structured configs to python path - If there is any change on the
# directory name or python file name, this will break
struct_config = Path(__file__).parent / (Path(__file__).with_suffix("").name + "_confs")
sys.path.append(str(struct_config))
from train_test_simple_net_confs.structured_confs import (
    ExperimentConfig,
    register_struct_configs,
)

register_struct_configs()

log = Logger(__name__).log


@dataclass
class DatasetContainer:
    train_dataset: JointsDataset1
    test_dataset: JointsDataset1
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    input_normalizer: Normalizer
    output_normalizer: Normalizer


def load_data(cfg: ExperimentConfig) -> DatasetContainer:
    train_paths = [Path(p) for p in cfg.path_config.train_paths]
    test_paths = [Path(p) for p in cfg.path_config.test_paths]

    train_dataset = JointsDataset1(
        train_paths,
        mode=cfg.dataset_config.dataset_type,
        include_prev_measured=cfg.dataset_config.include_prev_measured,
    )

    # Create input output normalizers
    input_normalizer = Normalizer(train_dataset.X)
    input_normalizer.to_json(Path(cfg.output_path) / "input_normalizer.json")
    train_dataset.set_input_normalizer(input_normalizer)

    output_normalizer = Normalizer(train_dataset.Y)
    output_normalizer.to_json(Path(cfg.output_path) / "output_normalizer.json")
    train_dataset.set_output_normalizer(output_normalizer)

    test_dataset = JointsDataset1(
        test_paths,
        mode=cfg.dataset_config.dataset_type,
        include_prev_measured=cfg.dataset_config.include_prev_measured,
    )
    test_dataset.set_input_normalizer(input_normalizer)
    test_dataset.set_output_normalizer(output_normalizer)

    with open(Path(cfg.output_path) / "dataset_info.json", "w") as file:
        json.dump(
            {
                "dataset_type": cfg.dataset_config.dataset_type,
                "include_prev_measured": cfg.dataset_config.include_prev_measured,
                "train_size": len(train_dataset),
                "test_size": len(test_dataset),
            },
            file,
            indent=4,
        )

    log.info(f"Train dataset size: {len(train_dataset)}")
    log.info(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train_config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.test_config.batch_size, shuffle=True
    )

    return DatasetContainer(
        train_dataset,
        test_dataset,
        train_dataloader,
        test_dataloader,
        input_normalizer,
        output_normalizer,
    )


def create_model(cfg: ExperimentConfig, dataset_container: DatasetContainer):
    in_features = dataset_container.train_dataset.get_input_dim()
    model = BestMLP2(in_features=in_features)

    if cfg.global_device:
        model = model.cuda()

    return model


def train_model(
    cfg: ExperimentConfig, dataset_container: DatasetContainer, model: BestMLP2
):
    output_path = Path(cfg.output_path)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_config.lr)

    loss_metric = torch.nn.MSELoss()

    trainer = Trainer(
        dataset_container.train_dataloader,
        dataset_container.test_dataloader,
        model,
        optimizer,
        loss_metric,
        cfg.train_config.epochs,
        output_path,
        cfg.global_device,
        True,
        cfg.train_config.log_interval,
    )
    trainer.train_loop()
    trainer.save_model("final_weights.pth")

    return trainer


def show_training_plots(trainer: Trainer):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(trainer.train_epoch_loss_list, label="train")
    ax[1].plot(trainer.valid_epoch_loss_list, label="valid")
    ax[1].axhline(
        y=trainer.train_epoch_loss_list[-1],
        color="r",
        linestyle="--",
        label="best training",
    )
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()

    plt.show()


def load_best_weights(model: BestMLP2, cfg: ExperimentConfig):
    log.info(f"Loading weights from {cfg.output_path}")
    model.load_state_dict(torch.load(Path(cfg.output_path) / "final_weights.pth"))
    return model


def calculate_test_metrics(
    cfg: ExperimentConfig, model: BestMLP2, dataset_container: DatasetContainer
) -> MetricsCalculator:
    model.eval()

    with torch.no_grad():
        pred_offsets = []
        gt_offsets = []
        input_jp = []

        input_normalizer = dataset_container.input_normalizer
        out_normalizer = dataset_container.output_normalizer

        for X, Y in dataset_container.test_dataloader:
            X = X.cuda()
            Y = Y.cuda()
            Y_hat = model(X)

            X = input_normalizer.reverse(X.cpu())
            Y_hat = out_normalizer.reverse(Y_hat.cpu())
            Y = out_normalizer.reverse(Y.cpu())

            pred_offsets.append(Y_hat.numpy())
            gt_offsets.append(Y.numpy())
            input_jp.append(X.numpy()[:, :6])  # Only joint positions

    input_jp = np.concatenate(input_jp, axis=0)
    pred_offsets = np.concatenate(pred_offsets, axis=0)
    gt_offsets = np.concatenate(gt_offsets, axis=0)

    metrics_calc = MetricsCalculator(
        experiment_name=cfg.dataset_config.dataset_type,
        input_jp=input_jp,
        gt_offset=gt_offsets,
        pred_offset=pred_offsets,
    )
    return metrics_calc


@hydra.main(
    version_base=None,
    config_path="./train_test_simple_net_confs/",
    config_name="train_test_simple_net",
)
def main(cfg: ExperimentConfig):
    log.info("Config file")
    print(OmegaConf.to_yaml(cfg))

    dataset_container = load_data(cfg)
    model = create_model(cfg, dataset_container)

    if cfg.actions.train:
        trainer = train_model(cfg, dataset_container, model)
        show_training_plots(trainer)

    if cfg.actions.test:
        model = load_best_weights(model, cfg)
        calc_metrics = calculate_test_metrics(cfg, model, dataset_container)
        exp_metrics = calc_metrics.get_metrics_container()
        exp_metrics.to_table().print()

        # Metrics with no corrections
        calc_metrics2 = MetricsCalculator(
            calc_metrics.experiment_name,
            calc_metrics.input_jp,
            calc_metrics.gt_offset,
            np.zeros_like(calc_metrics.gt_offset),
        )
        exp_metrics2 = calc_metrics2.get_metrics_container()
        log.info(f"Error with no corrections - {exp_metrics2.experiment_name}")
        exp_metrics2.to_table(with_jp=False).print(floatfmt=".5f")

    # from hydra.core.hydra_config import HydraConfig
    # print(HydraConfig.get().job.config_name)
    # print(HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
