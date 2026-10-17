"""PyTorch Dataset wrapping pre-computed feature matrices and targets."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MolecularDataset(Dataset):
    """
    Parameters
    ----------
    features : (N, D) feature matrix.
    targets : (N,) target values (normalised or raw).
    smiles : optional SMILES strings, kept for traceability.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        smiles: Optional[List[str]] = None,
    ) -> None:
        if len(features) != len(targets):
            raise ValueError(
                f"Features ({len(features)}) and targets ({len(targets)}) must have the same length."
            )
        if smiles is not None and len(smiles) != len(features):
            raise ValueError("smiles must have the same length as features.")

        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        self.smiles = list(smiles) if smiles is not None else [""] * len(features)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features[idx],
            "target": self.targets[idx],
            "smiles": self.smiles[idx],
        }

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]
