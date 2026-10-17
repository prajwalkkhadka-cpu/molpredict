"""Training loop: AdamW, cosine schedule with warmup, early stopping, checkpointing."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from molpredict.evaluation.metrics import MolecularMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Parameters
    ----------
    min_delta : minimum decrease in validation loss that counts as improvement.
    gradient_clip : max gradient norm (``None``/0 disables clipping).
    warmup_epochs : linear LR warm-up epochs before the cosine decay.
    min_lr : floor of the cosine schedule.
    early_stopping : enable/disable early stopping.
    target_scaler : fitted scaler; metrics are reported in ORIGINAL units.
    checkpoint_extra : extra picklable state saved in the checkpoint (feature
        scaler, target scaler, config...) so inference can be reproduced.

    After :meth:`fit`, the model holds the weights of the BEST epoch (not the last).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        checkpoint_dir: str = "checkpoints/",
        patience: int = 10,
        min_delta: float = 0.0,
        gradient_clip: Optional[float] = 1.0,
        warmup_epochs: int = 0,
        min_lr: float = 1e-6,
        early_stopping: bool = True,
        device: str = "auto",
        target_scaler=None,
        checkpoint_extra: Optional[Dict] = None,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_clip = gradient_clip
        self.early_stopping = early_stopping
        self.target_scaler = target_scaler
        self.checkpoint_extra = checkpoint_extra or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        floor = min_lr / learning_rate if learning_rate > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            span = max(1, epochs - warmup_epochs)
            progress = min(1.0, (epoch - warmup_epochs) / span)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.best_val_loss = float("inf")
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch = 0
        self.history: List[Dict[str, float]] = []

    # ---------------------------------------------------------------- epochs
    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            x = batch["features"].to(self.device)
            y = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            if self.gradient_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            self.optimizer.step()

            total += loss.item() * len(y)
            n += len(y)
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Returns (loss on normalised scale, metrics in original units)."""
        self.model.eval()
        total, n = 0.0, 0
        preds, targets = [], []
        for batch in loader:
            x = batch["features"].to(self.device)
            y = batch["target"].to(self.device)
            p = self.model(x)
            total += self.criterion(p, y).item() * len(y)
            n += len(y)
            preds.append(p.cpu().numpy())
            targets.append(y.cpu().numpy())

        y_p = np.concatenate(preds)
        y_t = np.concatenate(targets)
        if self.target_scaler is not None:
            y_p = self.target_scaler.inverse_transform(y_p.reshape(-1, 1)).ravel()
            y_t = self.target_scaler.inverse_transform(y_t.reshape(-1, 1)).ravel()
        return total / max(1, n), MolecularMetrics.compute_all(y_t, y_p)

    # ------------------------------------------------------------------- fit
    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_metrics": val_metrics,
        }
        payload.update(self.checkpoint_extra)
        torch.save(payload, self.checkpoint_dir / "best_model.pt")

    def fit(self) -> Dict[str, float]:
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            logger.info(
                "Epoch %03d/%03d | train loss %.4f | val RMSE %.4f | val R2 %.4f",
                epoch, self.epochs, train_loss, val_metrics["rmse"], val_metrics["r2"],
            )

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics
                self.best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(epoch, val_metrics)
            else:
                patience_counter += 1
                if self.early_stopping and patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d (best epoch: %d)", epoch, self.best_epoch)
                    break

        # Restore best weights so later evaluation (e.g. on the test set) uses them.
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.best_metrics
