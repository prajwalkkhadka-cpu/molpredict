"""
Feature engineering: Morgan fingerprints + a few 2D physicochemical descriptors.

Note: Crippen ``MolLogP`` is deliberately NOT used as an input descriptor. The
prediction target is LogP, so feeding an RDKit LogP estimate to the model would
let it mostly copy that estimate, and it would make the "model vs RDKit
baseline" comparison meaningless.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTORS = [
    "MolWt",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
    "NumAromaticRings",
    "FractionCSP3",
    "RingCount",
]


class FeatureEngineer:
    """
    Parameters
    ----------
    radius : Morgan radius (2 = ECFP4).
    n_bits : fingerprint length.
    use_descriptors : append 2D descriptors to the fingerprint.
    normalize_descriptors : z-score descriptors (scaler fitted on TRAIN only).
    descriptor_names : RDKit ``Descriptors`` function names to compute.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_descriptors: bool = True,
        normalize_descriptors: bool = True,
        descriptor_names: Optional[List[str]] = None,
    ) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self.use_descriptors = use_descriptors
        self.normalize_descriptors = normalize_descriptors
        self.descriptor_names = list(descriptor_names or DEFAULT_DESCRIPTORS)

        for name in self.descriptor_names:
            if not hasattr(Descriptors, name):
                raise ValueError(f"Unknown RDKit descriptor: {name}")

        self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        self._desc_mean: Optional[np.ndarray] = None
        self._desc_scale: Optional[np.ndarray] = None

    # ----------------------------------------------------------- properties
    @property
    def is_scaler_fitted(self) -> bool:
        return self._desc_mean is not None

    @property
    def feature_dim(self) -> int:
        return self.n_bits + (len(self.descriptor_names) if self.use_descriptors else 0)

    # --------------------------------------------------------- single mol
    def mol_to_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        fp = self._fp_gen.GetFingerprint(mol)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def mol_to_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        values = []
        for name in self.descriptor_names:
            try:
                val = float(getattr(Descriptors, name)(mol))
            except Exception:
                val = 0.0
            values.append(val if np.isfinite(val) else 0.0)
        return np.array(values, dtype=np.float32)

    def smiles_to_morgan_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        mol = Chem.MolFromSmiles(smiles)
        return None if mol is None else self.mol_to_fingerprint(mol)

    def compute_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        mol = Chem.MolFromSmiles(smiles)
        return None if mol is None else self.mol_to_descriptors(mol)

    # -------------------------------------------------------------- batch
    def extract_features(
        self, smiles_list: List[str], fit_scaler: bool = False
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Returns ``(X, valid_indices)``. ``valid_indices`` are positions in
        ``smiles_list`` that parsed successfully; ``X`` has one row per index.
        """
        fps, descs, valid = [], [], []
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                continue
            fps.append(self.mol_to_fingerprint(mol))
            if self.use_descriptors:
                descs.append(self.mol_to_descriptors(mol))
            valid.append(idx)

        if not valid:
            return np.zeros((0, self.feature_dim), dtype=np.float32), []

        X_fp = np.stack(fps).astype(np.float32)
        if not self.use_descriptors:
            return X_fp, valid

        X_desc = np.stack(descs).astype(np.float32)
        if self.normalize_descriptors:
            if fit_scaler:
                self._desc_mean = X_desc.mean(axis=0)
                std = X_desc.std(axis=0)
                self._desc_scale = np.where(std < 1e-8, 1.0, std)
            elif not self.is_scaler_fitted:
                raise RuntimeError(
                    "Descriptor scaler is not fitted. Call extract_features(..., "
                    "fit_scaler=True) on the training split first (or load_state())."
                )
            X_desc = (X_desc - self._desc_mean) / self._desc_scale

        return np.hstack([X_fp, X_desc]).astype(np.float32), valid

    # ------------------------------------------------------- persistence
    def get_state(self) -> Dict:
        """JSON/tensor-free state needed to reproduce features at inference."""
        return {
            "radius": self.radius,
            "n_bits": self.n_bits,
            "use_descriptors": self.use_descriptors,
            "normalize_descriptors": self.normalize_descriptors,
            "descriptor_names": self.descriptor_names,
            "desc_mean": None if self._desc_mean is None else self._desc_mean.tolist(),
            "desc_scale": None if self._desc_scale is None else self._desc_scale.tolist(),
        }

    @classmethod
    def from_state(cls, state: Dict) -> "FeatureEngineer":
        eng = cls(
            radius=state["radius"],
            n_bits=state["n_bits"],
            use_descriptors=state["use_descriptors"],
            normalize_descriptors=state["normalize_descriptors"],
            descriptor_names=state["descriptor_names"],
        )
        if state.get("desc_mean") is not None:
            eng._desc_mean = np.asarray(state["desc_mean"], dtype=np.float32)
            eng._desc_scale = np.asarray(state["desc_scale"], dtype=np.float32)
        return eng
