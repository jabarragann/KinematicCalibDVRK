from abc import ABC, abstractclassmethod
import json
from pathlib import Path
import numpy as np
import time
from rich.progress import track
from dataclasses import dataclass, field
from typing import Callable, List

# Torch
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from kincalib.utils.Logger import Logger


log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")


@dataclass
class Trainer:
    train_loader: DataLoader
    valid_loader: DataLoader
    net: nn.Module
    optimizer: nn.Module
    loss_metric: nn.Module
    epochs: int
    output_path: Path
    gpu_boole: bool = True
    save: bool = True
    log_interval: int = 1
    verbose = True

    def __post_init__(self):
        self.init_epoch = 0
        self.final_epoch = 0
        self.batch_count = 0
        self.scheduler = None

        self.train_iteration_loss_list: List[float] = []
        self.train_epoch_loss_list: List[float] = []
        self.valid_epoch_loss_list: List[float] = []

    def train_loop(self, verbose=True):
        self.verbose = verbose

        local_loss_sum = 0
        local_total = 0
        for epoch in track(range(self.init_epoch, self.epochs), "Training network"):
            time1 = time.time()

            epoch_loss_sum = 0
            epoch_total = 0

            for batch_idx, (x, y) in enumerate(self.train_loader):
                if self.gpu_boole:
                    x = x.cuda()
                    y = y.cuda()

                # loss calculation and gradient update:
                self.optimizer.zero_grad()
                outputs = self.net(x)
                loss = self.loss_metric(outputs, y)  # REMEMBER loss(OUTPUTS,LABELS)
                loss.backward()
                self.optimizer.step()  # Update parameters

                self.batch_count += 1
                # iteration loss
                local_loss_sum += loss * y.shape[0]
                local_total += y.shape[0]
                # epoch loss
                epoch_loss_sum += loss * y.shape[0]
                epoch_total += y.shape[0]

                if batch_idx % self.log_interval == 0 and batch_idx > 0:
                    local_loss = (local_loss_sum / local_total).cpu().item()
                    self.train_iteration_loss_list.append(local_loss)
                    local_loss = 0
                    local_total = 0

            # End of epoch statistics
            train_loss = epoch_loss_sum / epoch_total
            train_loss = train_loss.cpu().item()
            self.train_epoch_loss_list.append(train_loss)

            # Valid loss
            valid_loss = self.calculate_loss(self.valid_loader)
            self.valid_epoch_loss_list.append(valid_loss)

            # Print epoch information
            time2 = time.time()
            # if verbose:
            #     log.info(f"*" * 30)
            #     log.info(f"Epoch {epoch}/{self.epochs-1}:")
            #     log.info(f"Elapsed time for epoch: { time2 - time1:0.04f} s")
            #     log.info(f"Training loss:     {train_loss:0.8f}")
            #     log.info(f"END OF EPOCH METRICS")

        return train_loss, valid_loss

    def save_model(self, filename: str):
        torch.save(self.net.state_dict(), self.output_path / filename)

    def calculate_loss(self, dataloader: DataLoader):
        loss_sum = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                if self.gpu_boole:
                    x = x.cuda()
                    y = y.cuda()
                outputs = self.net(x)
                loss_sum += self.loss_metric(outputs, y) * y.shape[0]
                total += y.shape[0]

            loss = loss_sum / total
        return loss.cpu().data.item()
