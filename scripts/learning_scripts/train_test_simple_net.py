from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from natsort import natsorted
import numpy as np
from typing import List, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
from kincalib.Hydra import HydraConfig
from kincalib.Learning import Trainer, BestMLP2
from kincalib.utils.Logger import Logger
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from kincalib.Learning.Dataset import JointsDataset1, Normalizer
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig

log = Logger(__name__).log


@dataclass
class ExperimentConfig:
    global_device: bool
    output_path: Path
    path_config: PathConfig
    actions: Actions
    train_config: TrainConfig


@dataclass
class PathConfig:
    workspace: str
    train_paths: List[str]
    test_paths: List[str]


@dataclass
class Actions:
    train: bool
    test: bool


class TrainConfig:
    batch_size: int
    epochs: int
    log_interval: int


@dataclass
class DatasetContainer:
    train_dataset: JointsDataset1
    test_dataset: JointsDataset1
    train_dataloader: DataLoader
    test_dataloader: DataLoader


def load_data(cfg: ExperimentConfig) -> DatasetContainer:
    train_paths = [Path(p) for p in cfg.path_config.train_paths]
    test_paths = [Path(p) for p in cfg.path_config.test_paths]

    train_dataset = JointsDataset1(train_paths)
    normalizer = Normalizer(train_dataset.X)
    normalizer.to_json(Path(cfg.output_path) / "normalizer.json")
    train_dataset.set_normalizer(normalizer)

    test_dataset = JointsDataset1(test_paths)
    test_dataset.set_normalizer(normalizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train_config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.test_config.batch_size, shuffle=True
    )

    return DatasetContainer(
        train_dataset, test_dataset, train_dataloader, test_dataloader
    )


def train_model(cfg: ExperimentConfig, dataset_container: DatasetContainer):
    output_path = Path(cfg.output_path)
    model = BestMLP2()
    model.train()
    learning_rate = 0.00094843454
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if cfg.global_device:
        net = model.cuda()

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


# TODO: This triggers an error. Learn more about Omega conf
# cs = ConfigStore.instance()
# cs.store(name="base_config", node=ExperimentConfig)


@hydra.main(
    version_base=None,
    config_path="./train_test_simple_net_confs/",
    config_name="train_test_simple_net",
)
def main(cfg: ExperimentConfig):
    log.info("Config file")
    print(OmegaConf.to_yaml(cfg))

    dataset_container = load_data(cfg)
    # model = create_model(cfg, dataset_container)

    if cfg.actions.train:
        trainer = train_model(cfg, dataset_container)
        show_training_plots(trainer)

    # from hydra.core.hydra_config import HydraConfig
    # print(HydraConfig.get().job.config_name)
    # print(HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
