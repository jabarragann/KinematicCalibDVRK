import torch
from torch import nn


class BestMLP1(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_features, 140))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.00929993824427144))
        layers.append(nn.Linear(140, 6))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class BestMLP2(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, 16))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.00929993824427144))
        layers.append(nn.Linear(16, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.00929993824427144))
        layers.append(nn.Linear(32, 6))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    from kincalib.utils.Logger import Logger
    from kincalib.Learning import JointsDataset1, Normalizer
    from torch.utils.data import DataLoader
    from pathlib import Path

    log = Logger(__name__).log

    exp_root = []
    exp_root.append(
        "./data/experiments/data_collection1/08-11-2023-19-33-54/filtered_data.csv"
    )
    exp_root.append(
        "./data/experiments/data_collection1/08-11-2023-19-52-14/filtered_data.csv"
    )
    exp_root = [Path(p) for p in exp_root]

    train_data = JointsDataset1(path_lists=exp_root)
    normalizer = Normalizer(train_data.X)
    train_data.set_input_normalizer(normalizer)

    dataloader = DataLoader(train_data, batch_size=4, shuffle=True)

    model = BestMLP2(in_features=6)
    log.info("Test model with random data")
    log.info(model)
    log.info(model(torch.randn(1, 6)))

    log.info("Test model with dataloader")
    bx, by = next(iter(dataloader))
    log.info(model(bx))
