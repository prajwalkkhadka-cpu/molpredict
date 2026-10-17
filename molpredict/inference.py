"""Load a trained checkpoint and predict LogP for new SMILES."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from molpredict.features.engineering import FeatureEngineer
from molpredict.models.mlp import MolecularMLP


class Predictor:
    """Reproduces the exact training-time featurisation and target scaling."""

    def __init__(
        self,
        model: MolecularMLP,
        engineer: FeatureEngineer,
        target_mean: float,
        target_scale: float,
        device: str = "cpu",
        meta: Optional[Dict] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.engineer = engineer
        self.target_mean = float(target_mean)
        self.target_scale = float(target_scale)
        self.meta = meta or {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "Predictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Train a model first: python scripts/train.py"
            )
        ckpt = torch.load(path, map_location=device, weights_only=True)
        required = {"model_state_dict", "model_config", "feature_state", "target_mean", "target_scale"}
        missing = required - set(ckpt)
        if missing:
            raise ValueError(
                f"Checkpoint {path} is missing {sorted(missing)}; it was likely produced by an "
                "older version. Retrain with scripts/train.py."
            )
        model = MolecularMLP(**ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"])
        engineer = FeatureEngineer.from_state(ckpt["feature_state"])
        meta = {"epoch": ckpt.get("epoch"), "val_metrics": ckpt.get("val_metrics"),
                "target_col": ckpt.get("target_col", "logP")}
        return cls(model, engineer, ckpt["target_mean"], ckpt["target_scale"], device, meta)

    @torch.no_grad()
    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Predicted values in original units; ``None`` for unparseable SMILES."""
        X, valid = self.engineer.extract_features(list(smiles_list), fit_scaler=False)
        out: List[Optional[float]] = [None] * len(smiles_list)
        if not valid:
            return out
        y = self.model(torch.as_tensor(X, dtype=torch.float32, device=self.device)).cpu().numpy()
        y = y * self.target_scale + self.target_mean
        for i, v in zip(valid, np.atleast_1d(y)):
            out[i] = float(v)
        return out

    def predict_one(self, smiles: str) -> Optional[float]:
        return self.predict([smiles])[0]
