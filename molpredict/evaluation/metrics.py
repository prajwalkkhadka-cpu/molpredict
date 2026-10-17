"""Regression metrics for molecular property prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MolecularMetrics:
    """RMSE, MAE, R2, Pearson r and Spearman rho."""

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
        if len(y_true) == 0:
            raise ValueError("Cannot compute metrics on empty arrays.")

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        # R2 is undefined for < 2 samples; report NaN instead of a misleading number.
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

        if len(y_true) > 2 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
            pearson_r = float(stats.pearsonr(y_true, y_pred)[0])
            spearman_rho = float(stats.spearmanr(y_true, y_pred)[0])
        else:
            pearson_r = spearman_rho = float("nan")

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "pearson_r": round(pearson_r, 4),
            "spearman_rho": round(spearman_rho, 4),
        }
