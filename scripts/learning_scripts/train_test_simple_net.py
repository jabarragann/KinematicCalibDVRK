from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt
import torch
from kincalib.Learning import Trainer, BestMLP2
from kincalib.utils.Logger import Logger
from omegaconf import OmegaConf
import hydra
from kincalib.Learning.Dataset import JointsDataset1, Normalizer
from torch.utils.data import DataLoader

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
        train_dataset, test_dataset, train_dataloader, test_dataloader
    )


def train_model(cfg: ExperimentConfig, dataset_container: DatasetContainer):
    output_path = Path(cfg.output_path)
    in_features = dataset_container.train_dataset.get_input_dim()
    model = BestMLP2(in_features=in_features)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_config.lr)

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
