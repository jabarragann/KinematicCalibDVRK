from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
from hydra.core.config_store import ConfigStore


def register_struct_configs():
    """Enable hydra config validation using struct configs.

    Followed the example: https://hydra.cc/docs/tutorials/structured_config/schema/
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=ExperimentConfig)
    cs.store(group="train_config", name="base_train_config", node=TrainConfig)


@dataclass
class ExperimentConfig:
    global_device: str
    output_path: Path
    path_config: PathConfig
    dataset_config: DatasetConfig
    actions: Actions
    train_config: TrainConfig
    test_config: TestConfig


@dataclass
class DatasetConfig:
    dataset_type: str  # measured-setpoint or actual-measured
    include_prev_measured: bool


@dataclass
class PathConfig:
    workspace: Path
    datapath: Path
    handeye_path: Path
    train_paths: List[Path]
    test_paths: List[Path]


@dataclass
class Actions:
    train: bool
    test: bool


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    log_interval: int


@dataclass
class TestConfig:
    batch_size: int
