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
from kincalib.utils.Logger import Logger
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from kincalib.Learning.Dataset import JointsDataset1, Normalizer

log = Logger(__name__).log


@dataclass
class ExperimentConfig:
    path_config: PathConfig
    actions: Actions


@dataclass
class PathConfig:
    workspace: str
    train_paths: List[str]
    test_paths: List[str]


@dataclass
class Actions:
    train: bool
    test: bool


@dataclass
class DatasetContainer:
    train_dataset: JointsDataset1
    test_dataset: JointsDataset1


cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)


def load_data(cfg: ExperimentConfig) -> DatasetContainer:
    train_paths = [Path(p) for p in cfg.path_config.train_paths]
    test_paths = [Path(p) for p in cfg.path_config.test_paths]

    train_dataset = JointsDataset1(train_paths)
    test_dataset = JointsDataset1(test_paths)

    return DatasetContainer(train_dataset, test_dataset)


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

    # if cfg.actions.show_images:
    #     show_images(dataset_container.dl_train, dataset_container.label_parser)

    # if cfg.actions.train:
    #     model = train_with_image_dataset(cfg, dataset_container, model)

    # if cfg.actions.test:
    #     test_model(cfg, dataset_container, model)

    from hydra.core.hydra_config import HydraConfig

    print(HydraConfig.get().job.config_name)


if __name__ == "__main__":
    main()
