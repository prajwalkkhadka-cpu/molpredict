"""
MolPredict — molecular property (LogP) prediction from SMILES.

Pipeline: SMILES -> RDKit features (Morgan fingerprint + descriptors) -> MLP.
"""

__version__ = "1.1.0"

from molpredict.data.preprocessing import DataPreprocessor
from molpredict.evaluation.metrics import MolecularMetrics
from molpredict.features.engineering import FeatureEngineer
from molpredict.inference import Predictor
from molpredict.models.mlp import MolecularMLP
from molpredict.training.trainer import Trainer

__all__ = [
    "DataPreprocessor",
    "FeatureEngineer",
    "MolecularMLP",
    "MolecularMetrics",
    "Predictor",
    "Trainer",
]
