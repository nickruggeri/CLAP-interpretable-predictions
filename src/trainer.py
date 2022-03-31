import logging.handlers
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .architecture.clap import CLAP
from .data.utils import SubsetSampler
from .losses import accuracy, bernoulli_reconstruction_loss, latent_kl_divergence

list_base_loss = [
    "total_loss",
    "reconstruction_loss_pred",
    "reconstruction_loss_cl",
    "prediction_loss",
]
list_z_loss = [
    "prior_z_core_y_cl",
    "prior_z_style_cl",
    "prior_z_core_pred",
    "prior_z_style_pred",
]


class CLAPTrainer:
    def __init__(
        self,
        model: CLAP,
        train_dataset: Dataset,
        test_dataset: Dataset,
        optimizer: Optimizer,
        batch_size: int,
        save_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_workers = 1

        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.dev_mean = torch.tensor(self.train_dataset.mean, device=self.device)[
            None, :, None, None
        ]
        self.dev_std = torch.tensor(self.train_dataset.std, device=self.device)[
            None, :, None, None
        ]
        assert self.dev_mean.shape == self.dev_std.shape
        assert (
            self.dev_mean.shape[0]
            == self.dev_mean.shape[2]
            == self.dev_mean.shape[3]
            == 1
        )

        self.optimizer = optimizer

        # data loaders
        self.shuffle_data = False
        self.train_loader, self.test_loader = self.created_data_loaders()

        # logging
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = Path("../out") / datetime.now().strftime(
                "%d-%m-%Y_%H-%M-%S"
            )
        self.save_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Logging and saving path: {self.save_path}")
        logging.info(self.model)

        # loss functions
        self.reconstruction_loss = bernoulli_reconstruction_loss
        self.loss_y = torch.nn.BCEWithLogitsLoss()
        self.accuracy = accuracy

        # metrics and tensorboard logging
        log_dir = self.save_path / "LOG"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_metric_tracker = MetricTracker(
            list_base_loss + list_z_loss,
            train_val_flag="train",
            writer=self.writer,
        )

        self.list_val_metrics = [f"y_accuracy{i}" for i in range(model.n_outputs)] + [
            "y_accuracy"
        ]
        self.val_metric_tracker = MetricTracker(
            list_base_loss + self.list_val_metrics,
            train_val_flag="val",
            writer=self.writer,
        )

        # attributes utilized during training
        self.group_sparsity_reg: Optional[float] = None
        self.best_accuracy: float = 0.0
        self.beta_reg_values: Optional[torch.Tensor] = None
        self.y_reg_values: Optional[torch.Tensor] = None

    def train(
        self,
        y_reg: float,
        group_sparsity_reg: float = 0.0,
        beta_reg: float = 1.0,
        n_epochs_start: int = 200,
        n_epochs_beta: int = 400,
        n_epochs_end: int = 200,
        n_cycle_beta: int = 2,
    ) -> None:
        self.group_sparsity_reg = group_sparsity_reg
        self.best_accuracy = 0.0
        self.beta_reg_values, self.y_reg_values = cycle_params(
            beta_reg,
            y_reg,
            n_epochs_start,
            n_epochs_beta,
            n_epochs_end,
            n_cycle_beta,
        )  # cyclical annealing

        for epoch in range(n_epochs_start + n_epochs_beta + n_epochs_end):
            self.model.train()
            self.train_metric_tracker.reset()
            self.val_metric_tracker.reset()

            for batch in tqdm(self.train_loader, desc="Train"):
                x, y = batch

                self.model.zero_grad()
                out_dict = self.model(x, y)
                loss = self.calculate_loss(x, y, out_dict, epoch, validation=False)

                # break the training if any nan appears in the loss
                if torch.any(torch.isnan(loss)):
                    logging.warning(
                        "Training interrupted due to nan appearing during training."
                    )
                    torch.cuda.empty_cache()
                    return

                # update main model
                loss.backward()
                self.optimizer.step()

            self.validate(epoch)
            self.epoch_end(epoch)
            torch.cuda.empty_cache()

    def created_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_sampler = SubsetSampler(50000, len(self.train_dataset))
        test_sampler = SubsetSampler(10000, len(self.test_dataset))

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            sampler=train_sampler,
            shuffle=self.shuffle_data,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            shuffle=self.shuffle_data,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return train_loader, test_loader

    def calculate_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        out_dict: Dict[str, torch.Tensor],
        epoch: int,
        validation: bool = False,
    ) -> torch.Tensor:
        metric_tracker = (
            self.val_metric_tracker if validation else self.train_metric_tracker
        )

        # Reconstruction loss
        rec_loss_pred = (
            self.reconstruction_loss(
                out_dict["pred"]["x_reconstructed"],
                x * self.dev_std + self.dev_mean,
            )
            / 2
        )
        rec_loss_cl = (
            self.reconstruction_loss(
                out_dict["cl"]["x_reconstructed"],
                x * self.dev_std + self.dev_mean,
            )
            / 2
        )
        metric_tracker.update("reconstruction_loss_pred", rec_loss_pred.item())
        metric_tracker.update("reconstruction_loss_cl", rec_loss_pred.item())

        reconstruction_loss = rec_loss_pred + rec_loss_cl

        # Prediction loss
        y_reg_epoch = self.y_reg_values[epoch]
        if y_reg_epoch > 0:
            y_pred = out_dict["pred"]["y_pred"]
            loss_y = self.loss_y(y_pred, y.type_as(y_pred)) / 2
        else:
            loss_y = torch.tensor(0, dtype=torch.float, device=self.device)
        metric_tracker.update("prediction_loss", loss_y.item())

        # calculate kl div / loss on prior
        beta_reg_epoch = self.beta_reg_values[epoch]
        if not validation and beta_reg_epoch > 0:
            total_z_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            for name_z_loss in list_z_loss:
                tmp_z_loss = latent_kl_divergence(
                    name_z_loss, out_dict, self.model.z_core_dim, self.model.z_style_dim
                )
                total_z_loss += tmp_z_loss
                metric_tracker.update(name_z_loss, tmp_z_loss.item())
        else:
            total_z_loss = torch.tensor(0, dtype=torch.float, device=self.device)

        # calculate sparsity regularization on prediction and decoder weights relative to z_core
        if self.group_sparsity_reg > 0:
            pred_weights = self.model.get_prediction_weights()
            decoder_weights = self.model.get_decoder_first_linear_layer()
            decoder_weights = decoder_weights[
                :, : self.model.z_core_dim
            ]  # get only weights relative to z_core

            all_weights = torch.cat([pred_weights, decoder_weights], dim=0)
            group_sparsity = all_weights.norm(p="fro", dim=0).sum()
        else:
            group_sparsity = torch.tensor(0, dtype=torch.float, device=self.device)

        # calculate total loss
        total_loss = (
            reconstruction_loss
            + beta_reg_epoch * total_z_loss
            + y_reg_epoch * loss_y
            + self.group_sparsity_reg * group_sparsity
        )
        metric_tracker.update("total_loss", total_loss.item())

        return total_loss

    def validate(self, epoch: int) -> None:
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test"):
                x, y = batch

                out_dict = self.model(x, y)

                self.calculate_loss(
                    x, y, out_dict, epoch, validation=True
                )  # only for metric tracking here
                self.calculate_val_metrics(x, y, out_dict)

    def calculate_val_metrics(
        self, x: torch.Tensor, y: torch.Tensor, out_dict: Dict[str, torch.Tensor]
    ) -> None:
        y_pred = torch.sigmoid(out_dict["pred"]["y_pred"])
        single_label_acc, acc_mean = self.accuracy(y_pred, y)

        for i, metric in enumerate(self.list_val_metrics):
            if i < self.model.n_outputs:  # accuracy for everyone of the single labels
                self.val_metric_tracker.update(metric, single_label_acc[i].item())
            elif i == self.model.n_outputs:  # average accuracy out of all the labels
                self.val_metric_tracker.update(metric, acc_mean.item())
            if i > self.model.n_outputs:
                break

    def save_checkpoint(self, state: Any, is_best: bool) -> None:
        filename = self.save_path / "checkpoint.pth.tar.gz"
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.save_path / "model_best.pth.tar.gz")

    def epoch_end(self, epoch: int) -> None:
        # log epoch
        self.train_metric_tracker.update_avg(epoch)
        self.val_metric_tracker.update_avg(epoch)
        logging.info(f"Epoch: {epoch}")
        logging.info(f"Train:")
        for key, value in self.train_metric_tracker.result().items():
            logging.info("    {:15s}: {}".format(str(key), value))
        logging.info(f"Test:")
        for key, value in self.val_metric_tracker.result().items():
            logging.info("    {:15s}: {}".format(str(key), value))

        # checkpoint and update best_score
        epoch_avg_acc = self.val_metric_tracker._data.average["y_accuracy"]
        is_best = epoch_avg_acc > self.best_accuracy
        self.best_accuracy = max(epoch_avg_acc, self.best_accuracy)
        self.save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            is_best,
        )


# from https://github.com/victoresque/pytorch-template/blob/41dc06f6f8f3f38c6ed49f01ff7d89cd5688adc4/utils/util.py#L46
class MetricTracker:
    def __init__(
        self,
        keys: List[Any],
        train_val_flag: bool,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.writer = writer
        self.train_val_flag = train_val_flag
        if "zcore_cl_pred" in keys:
            keys.append("zcore_cl_pred_cl")
            keys.append("zcore_cl_pred_pred")
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self._data_final = {key: [] for key in keys}
        self.reset()

    def reset(self) -> None:
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: Any, value: Any, n: int = 1) -> None:
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_avg(self, n: int) -> None:
        for key in self._data_final:
            if self.writer is not None:
                value = self._data.average[key]
                self._data_final[key].append(value)
                self.writer.add_scalar(self.train_val_flag + key, value, n)

    def avg(self, key: Any) -> None:
        return self._data.average[key]

    def result(self) -> Dict[Any, Any]:
        return dict(self._data.average)


def frange_cycle_linear(
    start: int, stop: int, n_epoch: int, n_cycle: int = 4, ratio: float = 0.8
) -> torch.Tensor:
    """Create the schedules for cyclical annealing."""
    L = torch.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def cycle_params(
    beta: float,
    y_reg: float,
    n_start: int,
    n_epochs_beta: int,
    n_end: int,
    n_cycle_beta: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the cyclical annealing schedules for:
    - the beta parameter (KL divergence multiplier in the ELBO)
    - the y_reg parameter (prediction loss multiplier in the ELBO)
    """
    beta_reg_values = torch.ones(n_start + n_epochs_beta + n_end)
    beta_reg_values[0:n_start] = 0
    beta_reg_values[n_start : n_epochs_beta + n_start] = frange_cycle_linear(
        0, 1, n_epochs_beta, n_cycle_beta
    )
    beta_reg_values *= beta

    y_reg_values = torch.ones(n_start + n_epochs_beta + n_end)
    y_reg_values[0 : n_start + n_epochs_beta] = frange_cycle_linear(
        0, 1, n_epochs_beta + n_start, 1
    )
    y_reg_values *= y_reg
    return beta_reg_values, y_reg_values
