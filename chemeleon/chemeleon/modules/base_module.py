# pylint: disable=unused-argument
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch
from torchmetrics import MeanAbsoluteError


class BaseModule(LightningModule):
    def __init__(self, _config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        # optimizer
        self.optimizer = _config["optimizer"]
        self.lr = _config["lr"]
        self.weight_decay = _config["weight_decay"]
        self.scheduler = _config["scheduler"]
        self.patience = _config["patience"]

        # test
        self.test_step_outputs = []
        self.cond_scale = _config["cond_scale"]

        # metrics
        self.mae = MeanAbsoluteError()

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        ret = self.forward(batch)
        self._log_metrics(ret, "train", batch.num_graphs)
        return ret["loss"]

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        ret = self.forward(batch)
        self._log_metrics(ret, "val", batch.num_graphs)
        return ret["loss"]

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        return self._set_configure_optimizers()

    def _log_metrics(
        self, ret: dict[str, Any], split: str, batch_size: Optional[int] = None
    ) -> None:
        self.log(
            f"{split}/loss",
            ret["loss"],
            on_step=True if split == "train" else False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{split}/mae_lattice",
            self.mae(ret["pred_noise_lattice"], ret["true_noise_lattice"]),
            on_step=True if split == "train" else False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{split}/mae_coords",
            self.mae(ret["pred_noise_coords"], ret["true_noise_coords"]),
            on_step=True if split == "train" else False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        if "vb_loss_atom_types" in ret:
            self.log(
                f"{split}/vb_loss_atom_types",
                ret["vb_loss_atom_types"],
                on_step=True if split == "train" else False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        if "ce_loss_atom_types" in ret:
            self.log(
                f"{split}/ce_loss_atom_types",
                ret["ce_loss_atom_types"],
                on_step=True if split == "train" else False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _set_configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        lr = self.lr
        weight_decay = self.weight_decay
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        # get max_steps
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps
        # set scheduler
        if self.scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        elif self.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", min_lr=1e-6, factor=0.8, patience=self.patience
            )
        elif self.scheduler == "linear_decay":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, max_steps)
        else:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "learning rate",
            "monitor": "val/loss",
        }

        return ([optimizer], [lr_scheduler])
