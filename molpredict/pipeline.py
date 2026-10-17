"""End-to-end training pipeline: data -> features -> model -> evaluation -> checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from rdkit.Chem import Crippen
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from molpredict.chem import parse_smiles
from molpredict.data.dataset import MolecularDataset
from molpredict.data.preprocessing import DataPreprocessor
from molpredict.evaluation.metrics import MolecularMetrics
from molpredict.features.engineering import FeatureEngineer
from molpredict.models.mlp import MolecularMLP
from molpredict.training.trainer import Trainer

logger = logging.getLogger(__name__)

MIN_RELIABLE_DATASET = 500


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _featurize_split(engineer: FeatureEngineer, df, y_norm_fn, fit: bool):
    """Featurise a split and keep rows/targets perfectly aligned."""
    X, valid = engineer.extract_features(df["canonical_smiles"].tolist(), fit_scaler=fit)
    df = df.iloc[valid].reset_index(drop=True)
    return X, df


def run_training(cfg: Dict) -> Dict:
    """Train with the given config dict; returns a results dictionary."""
    d, f, m, t = cfg["data"], cfg["features"], cfg["model"]["mlp"], cfg["training"]
    seed = d["random_seed"]
    set_seed(seed)

    # ---- data ------------------------------------------------------------
    pre = DataPreprocessor(
        smiles_col=d["smiles_col"], target_col=d["target_col"],
        test_size=d["test_size"], val_size=d["val_size"],
        random_seed=seed, split_method=d.get("split_method", "scaffold"),
    )
    cleaned = pre.clean(pre.load(d["path"]))
    if len(cleaned) < MIN_RELIABLE_DATASET:
        logger.warning(
            "Only %d molecules: validation/test metrics will be very noisy. "
            "Use a larger dataset (e.g. MoleculeNet ESOL/Lipophilicity) for real results.",
            len(cleaned),
        )
    train_df, val_df, test_df = pre.split(cleaned)

    # ---- features (scaler fitted on TRAIN only) ----------------------------
    engineer = FeatureEngineer(
        radius=f["fingerprint"]["radius"], n_bits=f["fingerprint"]["n_bits"],
        use_descriptors=f["descriptors"]["enabled"],
        normalize_descriptors=f["descriptors"]["normalize"],
        descriptor_names=f["descriptors"].get("names"),
    )
    X_train, train_df = _featurize_split(engineer, train_df, None, fit=True)
    X_val, val_df = _featurize_split(engineer, val_df, None, fit=False)
    X_test, test_df = _featurize_split(engineer, test_df, None, fit=False)

    y_train, y_val, y_test = pre.normalize_targets(train_df, val_df, test_df)

    datasets = {
        "train": MolecularDataset(X_train, y_train, train_df["canonical_smiles"].tolist()),
        "val": MolecularDataset(X_val, y_val, val_df["canonical_smiles"].tolist()),
        "test": MolecularDataset(X_test, y_test, test_df["canonical_smiles"].tolist()),
    }
    bs = t["batch_size"]
    # BatchNorm cannot train on a final batch of size 1.
    drop_last = m["batch_norm"] and len(datasets["train"]) % bs == 1
    train_loader = DataLoader(datasets["train"], batch_size=bs, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(datasets["val"], batch_size=bs)
    test_loader = DataLoader(datasets["test"], batch_size=bs)

    # ---- model + trainer -----------------------------------------------------
    model = MolecularMLP(
        input_dim=datasets["train"].feature_dim,
        hidden_dims=m["hidden_dims"], dropout=m["dropout"],
        use_batch_norm=m["batch_norm"], activation=m["activation"],
    )
    es = t["early_stopping"]
    sched = t.get("scheduler", {})
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
        epochs=t["epochs"], checkpoint_dir=cfg["checkpoint"]["dir"],
        patience=es["patience"], min_delta=es.get("min_delta", 0.0),
        early_stopping=es.get("enabled", True),
        gradient_clip=t.get("gradient_clip"),
        warmup_epochs=sched.get("warmup_epochs", 0), min_lr=sched.get("min_lr", 1e-6),
        device=t.get("device", "auto"), target_scaler=pre.target_scaler,
        checkpoint_extra={
            "model_config": model.config,
            "feature_state": engineer.get_state(),
            "target_mean": float(pre.target_scaler.mean_[0]),
            "target_scale": float(pre.target_scaler.scale_[0]),
            "target_col": d["target_col"],
            "config": cfg,
        },
    )
    best_val = trainer.fit()
    _, test_metrics = trainer.evaluate(test_loader)   # uses BEST-epoch weights

    # ---- baselines on the same test molecules ------------------------------------
    y_test_true = test_df[d["target_col"]].values
    crippen = np.array([Crippen.MolLogP(parse_smiles(s)) for s in test_df["canonical_smiles"]])
    ridge = Ridge(alpha=1.0).fit(X_train, train_df[d["target_col"]].values)
    baselines = {
        "train_mean": MolecularMetrics.compute_all(
            y_test_true, np.full_like(y_test_true, train_df[d["target_col"]].mean(), dtype=float)),
        "ridge_same_features": MolecularMetrics.compute_all(y_test_true, ridge.predict(X_test)),
    }
    if d["target_col"].lower() == "logp":
        baselines["rdkit_crippen_logp"] = MolecularMetrics.compute_all(y_test_true, crippen)

    results = {
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "best_epoch": trainer.best_epoch,
        "val": best_val, "test": test_metrics, "baselines": baselines,
        "checkpoint": str(Path(cfg["checkpoint"]["dir"]) / "best_model.pt"),
    }
    out = Path(cfg["checkpoint"]["dir"]) / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    return results


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Train the MolPredict model")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)
    res = run_training(cfg)

    logger.info("Split sizes: train=%d val=%d test=%d (best epoch %d)",
                res["n_train"], res["n_val"], res["n_test"], res["best_epoch"])
    logger.info("Validation: %s", res["val"])
    logger.info("Test (MLP): %s", res["test"])
    for name, metrics in res["baselines"].items():
        logger.info("Test (baseline %-20s): %s", name, metrics)
    logger.info("Checkpoint saved to %s", res["checkpoint"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
