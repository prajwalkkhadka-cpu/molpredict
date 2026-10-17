"""
Data preprocessing: load -> clean -> split -> normalise targets.

Splitting defaults to a *scaffold* split (molecules sharing a Bemis-Murcko
scaffold stay in the same subset), which is the standard honest protocol for
molecular property prediction. A plain random split is available too.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Typographic minus signs that pandas would otherwise turn into NaN.
_MINUS_CHARS = ("\u2212", "\u2013", "\u2014", "\u2012")


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert to float, accepting unicode minus/dash characters."""
    if not pd.api.types.is_numeric_dtype(series):  # object OR pandas StringDtype
        series = series.astype(str).str.strip()
        for ch in _MINUS_CHARS:
            series = series.str.replace(ch, "-", regex=False)
    return pd.to_numeric(series, errors="coerce")


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES ('' for acyclic molecules)."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:  # pragma: no cover - defensive
        return ""


class DataPreprocessor:
    """
    Parameters
    ----------
    smiles_col, target_col : column names in the CSV.
    test_size : fraction of ALL data used for the test set.
    val_size : fraction of the *remaining* (train+val) data used for validation.
    random_seed : reproducibility seed.
    split_method : ``"scaffold"`` (default) or ``"random"``.
    """

    def __init__(
        self,
        smiles_col: str = "smiles",
        target_col: str = "logP",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_seed: int = 42,
        split_method: str = "scaffold",
    ) -> None:
        if split_method not in ("scaffold", "random"):
            raise ValueError("split_method must be 'scaffold' or 'random'")
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.split_method = split_method

        self.target_scaler = StandardScaler()
        self._is_fitted = False

    # ------------------------------------------------------------------ load
    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded %d rows from %s", len(df), path)
        for col in (self.smiles_col, self.target_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
        return df

    # ----------------------------------------------------------------- clean
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop missing/invalid rows, canonicalise SMILES, remove duplicates."""
        n_raw = len(df)
        df = df.copy()

        df = df.dropna(subset=[self.smiles_col, self.target_col])
        df[self.target_col] = _to_numeric(df[self.target_col])
        df = df.dropna(subset=[self.target_col])
        n_after_missing = len(df)

        canonical, keep = [], []
        for idx, smi in df[self.smiles_col].items():
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol is None:
                logger.warning("Invalid SMILES skipped: %r", smi)
                continue
            canonical.append(Chem.MolToSmiles(mol))
            keep.append(idx)
        df = df.loc[keep].copy()
        df["canonical_smiles"] = canonical
        n_valid = len(df)

        df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")

        logger.info(
            "Cleaning: %d -> %d rows (missing/non-numeric: %d, invalid SMILES: %d, duplicates: %d)",
            n_raw, len(df), n_raw - n_after_missing, n_after_missing - n_valid, n_valid - len(df),
        )
        return df.reset_index(drop=True)

    # ----------------------------------------------------------------- split
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if "canonical_smiles" not in df.columns:
            raise ValueError("Run clean() first: 'canonical_smiles' column missing.")
        if len(df) < 10:
            raise ValueError(f"Need at least 10 molecules to split, got {len(df)}.")

        if self.split_method == "scaffold":
            train, val, test = self._scaffold_split(df)
        else:
            train, val, test = self._random_split(df)

        logger.info("Split (%s) -> train: %d | val: %d | test: %d",
                    self.split_method, len(train), len(val), len(test))
        return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))

    def _scaffold_split(self, df: pd.DataFrame):
        groups: dict[str, List[int]] = defaultdict(list)
        for i, smi in enumerate(df["canonical_smiles"]):
            groups[murcko_scaffold(smi)].append(i)

        # Largest scaffold groups first -> they go to train; rare scaffolds
        # end up in val/test. Ties are broken deterministically with the seed.
        rng = np.random.RandomState(self.random_seed)
        sets = list(groups.values())
        rng.shuffle(sets)
        sets.sort(key=len, reverse=True)

        n = len(df)
        n_test = max(1, int(round(self.test_size * n)))
        n_val = max(1, int(round(self.val_size * (n - n_test))))
        n_train_target = n - n_test - n_val

        train_idx, val_idx, test_idx = [], [], []
        for s in sets:
            if len(train_idx) + len(s) <= n_train_target:
                train_idx += s
            elif len(val_idx) + len(s) <= n_val:
                val_idx += s
            else:
                test_idx += s
        if not val_idx or not test_idx or not train_idx:
            raise ValueError(
                "Scaffold split produced an empty subset (too few distinct scaffolds); "
                "use split_method='random' or more data."
            )
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    def _random_split(self, df: pd.DataFrame):
        def stratified_split(frame: pd.DataFrame, test_size: float):
            n_test = int(np.ceil(test_size * len(frame))) if test_size < 1 else int(test_size)
            # Stratify on target quantiles only when every bin can be represented.
            for q in (10, 5, 3, 2):
                bins = pd.qcut(frame[self.target_col].values, q=q, labels=False, duplicates="drop")
                counts = np.bincount(bins)
                if n_test >= len(counts) and (len(frame) - n_test) >= len(counts) and counts.min() >= 2:
                    return train_test_split(frame, test_size=test_size,
                                            stratify=bins, random_state=self.random_seed)
            return train_test_split(frame, test_size=test_size, random_state=self.random_seed)

        train_val, test = stratified_split(df, self.test_size)
        train, val = stratified_split(train_val, self.val_size)
        return train, val, test

    # ------------------------------------------------------- target scaling
    def normalize_targets(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the scaler on TRAIN targets only and transform all splits."""
        col = self.target_col
        y_train = self.target_scaler.fit_transform(train_df[[col]].values).ravel()
        y_val = self.target_scaler.transform(val_df[[col]].values).ravel()
        y_test = self.target_scaler.transform(test_df[[col]].values).ravel()
        self._is_fitted = True
        logger.info("Target normalisation: mean=%.4f std=%.4f",
                    self.target_scaler.mean_[0], self.target_scaler.scale_[0])
        return y_train, y_val, y_test

    def inverse_transform_targets(self, y_norm: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call normalize_targets() before inverse_transform.")
        return self.target_scaler.inverse_transform(np.asarray(y_norm).reshape(-1, 1)).ravel()

    def run(self, path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """load -> clean -> split."""
        return self.split(self.clean(self.load(path)))
