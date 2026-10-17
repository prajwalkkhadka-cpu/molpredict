"""Multi-layer perceptron for molecular feature vectors."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class MolecularMLP(nn.Module):
    """
    Plain feed-forward regressor: [Linear -> (BatchNorm) -> activation -> (Dropout)] x N -> Linear.

    Parameters
    ----------
    input_dim : feature dimension (fingerprint bits + descriptors).
    hidden_dims : units per hidden layer.
    output_dim : outputs (1 for single-property regression).
    dropout : dropout probability after each hidden block.
    use_batch_norm : apply BatchNorm1d after each linear layer.
    activation : 'relu', 'gelu' or 'silu'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims) if hidden_dims is not None else [1024, 512, 256, 128]
        if activation.lower() not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}")

        self.config: Dict = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "dropout": dropout,
            "use_batch_norm": use_batch_norm,
            "activation": activation.lower(),
        }
        act_fn = _ACTIVATIONS[activation.lower()]

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
