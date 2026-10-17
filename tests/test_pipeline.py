"""Tests for MolPredict. Each regression test is tied to a bug found in review."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from molpredict.chem import lipinski_rule_of_five, mol_properties, parse_smiles
from molpredict.data.dataset import MolecularDataset
from molpredict.data.preprocessing import DataPreprocessor, murcko_scaffold
from molpredict.evaluation.metrics import MolecularMetrics
from molpredict.features.engineering import FeatureEngineer
from molpredict.inference import Predictor
from molpredict.models.mlp import MolecularMLP
from molpredict.training.trainer import Trainer

SAMPLE = "data/sample_data.csv"


# ---------------------------------------------------------------- packaging
def test_package_imports():
    import molpredict  # previously crashed: imported a non-existent gnn module
    assert hasattr(molpredict, "Predictor")


# ----------------------------------------------------------------- features
def test_morgan_fingerprint_generation():
    eng = FeatureEngineer(radius=2, n_bits=1024, use_descriptors=False)
    fp = eng.smiles_to_morgan_fingerprint("CCO")
    assert fp is not None and len(fp) == 1024 and fp.sum() > 0


def test_invalid_smiles_returns_none():
    eng = FeatureEngineer(use_descriptors=False)
    assert eng.smiles_to_morgan_fingerprint("not a smiles") is None


def test_descriptors_do_not_include_target_leak():
    assert "MolLogP" not in FeatureEngineer().descriptor_names


def test_extract_features_alignment_with_invalid_smiles():
    eng = FeatureEngineer(n_bits=64, use_descriptors=True)
    X, valid = eng.extract_features(["CCO", "garbage", "c1ccccc1"], fit_scaler=True)
    assert valid == [0, 2]
    assert X.shape == (2, eng.feature_dim)


def test_unfitted_scaler_raises_instead_of_silently_skipping():
    eng = FeatureEngineer(n_bits=64)
    with pytest.raises(RuntimeError):
        eng.extract_features(["CCO"], fit_scaler=False)


def test_feature_state_roundtrip():
    eng = FeatureEngineer(n_bits=64)
    eng.extract_features(["CCO", "c1ccccc1", "CC(=O)O"], fit_scaler=True)
    clone = FeatureEngineer.from_state(eng.get_state())
    a, _ = eng.extract_features(["CCN"])
    b, _ = clone.extract_features(["CCN"])
    np.testing.assert_allclose(a, b, rtol=1e-6)


# ------------------------------------------------------------ preprocessing
def test_unicode_minus_targets_are_kept():
    """Regression: '−0.07' (U+2212) used to become NaN and the row was dropped."""
    df = pd.DataFrame({"smiles": ["CCO", "CC(=O)O"], "logP": ["\u22120.07", "\u22120.17"]})
    out = DataPreprocessor().clean(df)
    assert len(out) == 2
    assert out["logP"].tolist() == [-0.07, -0.17]


def test_sample_dataset_loses_no_rows_except_duplicates():
    pre = DataPreprocessor()
    raw = pre.load(SAMPLE)
    clean = pre.clean(raw)
    assert len(raw) - len(clean) <= 1  # the file contains one duplicate molecule


def test_clean_drops_invalid_and_duplicate():
    df = pd.DataFrame({"smiles": ["CCO", "OCC", "bad", None], "logP": [1.0, 1.0, 2.0, 3.0]})
    out = DataPreprocessor().clean(df)
    assert len(out) == 1


@pytest.mark.parametrize("method", ["scaffold", "random"])
def test_split_is_disjoint_and_complete(method):
    pre = DataPreprocessor(split_method=method)
    clean = pre.clean(pre.load(SAMPLE))
    tr, va, te = pre.split(clean)
    sets = [set(x["canonical_smiles"]) for x in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(tr) + len(va) + len(te) == len(clean)
    assert min(len(tr), len(va), len(te)) > 0


def test_scaffold_split_has_no_scaffold_leakage():
    pre = DataPreprocessor(split_method="scaffold")
    clean = pre.clean(pre.load(SAMPLE))
    tr, va, te = pre.split(clean)
    scaf = [{murcko_scaffold(s) for s in d["canonical_smiles"]} for d in (tr, va, te)]
    assert not (scaf[0] & scaf[1]) and not (scaf[0] & scaf[2]) and not (scaf[1] & scaf[2])


def test_random_split_works_on_tiny_data():
    """Regression: decile stratification crashed on small datasets."""
    smi = ["C" * n for n in range(1, 21)]
    df = pd.DataFrame({"smiles": smi, "logP": np.linspace(0, 5, 20)})
    pre = DataPreprocessor(split_method="random")
    tr, va, te = pre.split(pre.clean(df))
    assert len(tr) + len(va) + len(te) == 20


def test_target_scaler_fit_on_train_only():
    pre = DataPreprocessor()
    tr = pd.DataFrame({"logP": [0.0, 2.0]})
    va = pd.DataFrame({"logP": [100.0]})
    te = pd.DataFrame({"logP": [100.0]})
    y_tr, y_va, _ = pre.normalize_targets(tr, va, te)
    assert abs(pre.target_scaler.mean_[0] - 1.0) < 1e-9
    assert y_va[0] > 10
    np.testing.assert_allclose(pre.inverse_transform_targets(y_tr), [0.0, 2.0])


# ------------------------------------------------------------------- model
def test_mlp_forward_pass():
    model = MolecularMLP(input_dim=128, hidden_dims=[64, 32], dropout=0.1)
    assert model(torch.randn(8, 128)).shape == torch.Size([8])


def test_mlp_rejects_unknown_activation():
    with pytest.raises(ValueError):
        MolecularMLP(input_dim=4, activation="swishy")


def test_mlp_can_overfit_tiny_batch():
    torch.manual_seed(0)
    x, y = torch.randn(16, 32), torch.randn(16)
    model = MolecularMLP(input_dim=32, hidden_dims=[64], dropout=0.0, use_batch_norm=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


# ----------------------------------------------------------------- metrics
def test_metrics_computation():
    res = MolecularMetrics.compute_all([1, 2, 3, 4, 5], [1.1, 1.9, 3.2, 3.8, 5.1])
    assert res["r2"] > 0.95 and res["rmse"] < 0.2


def test_metrics_constant_prediction_is_nan_not_zero():
    res = MolecularMetrics.compute_all([1, 2, 3, 4], [2, 2, 2, 2])
    assert np.isnan(res["pearson_r"])


# --------------------------------------------------------------- chemistry
def test_lipinski_aspirin_passes():
    p = mol_properties(parse_smiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert abs(p["mw"] - 180.16) < 0.1
    _, violations = lipinski_rule_of_five(p["mw"], p["crippen_logp"], p["hbd"], p["hba"])
    assert violations == 0


def test_lipinski_flags_violations():
    _, v = lipinski_rule_of_five(mw=900, logp=7, hbd=8, hba=15)
    assert v == 4


def test_parse_smiles_rejects_garbage():
    assert parse_smiles("this is not smiles") is None
    assert parse_smiles("") is None
    assert parse_smiles(None) is None


# ------------------------------------------- train -> save -> load -> predict
def test_checkpoint_roundtrip_matches_training_time_predictions(tmp_path):
    torch.manual_seed(0)
    pre = DataPreprocessor(split_method="random")
    clean = pre.clean(pre.load(SAMPLE))
    tr, va, te = pre.split(clean)

    eng = FeatureEngineer(n_bits=128)
    X_tr, _ = eng.extract_features(tr["canonical_smiles"].tolist(), fit_scaler=True)
    X_va, _ = eng.extract_features(va["canonical_smiles"].tolist())
    y_tr, y_va, _ = pre.normalize_targets(tr, va, te)

    model = MolecularMLP(input_dim=eng.feature_dim, hidden_dims=[32], dropout=0.0)
    trainer = Trainer(
        model, DataLoader(MolecularDataset(X_tr, y_tr), batch_size=8, shuffle=True),
        DataLoader(MolecularDataset(X_va, y_va), batch_size=8),
        epochs=3, checkpoint_dir=str(tmp_path), target_scaler=pre.target_scaler, device="cpu",
        checkpoint_extra={
            "model_config": model.config, "feature_state": eng.get_state(),
            "target_mean": float(pre.target_scaler.mean_[0]),
            "target_scale": float(pre.target_scaler.scale_[0]), "target_col": "logP",
        },
    )
    trainer.fit()

    predictor = Predictor.from_checkpoint(tmp_path / "best_model.pt")
    smiles = te["canonical_smiles"].tolist()
    got = np.array(predictor.predict(smiles))

    # Reference: run the (best-weights-restored) in-memory model by hand.
    X_te, _ = eng.extract_features(smiles)
    trainer.model.eval()
    with torch.no_grad():
        ref = trainer.model(torch.tensor(X_te)).numpy()
    ref = pre.inverse_transform_targets(ref)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_predictor_returns_none_for_invalid_smiles(tmp_path):
    eng = FeatureEngineer(n_bits=32)
    eng.extract_features(["CCO", "c1ccccc1", "CCN"], fit_scaler=True)
    model = MolecularMLP(input_dim=eng.feature_dim, hidden_dims=[8])
    p = Predictor(model, eng, 0.0, 1.0)
    out = p.predict(["CCO", "garbage", "c1ccccc1"])
    assert out[1] is None and out[0] is not None and out[2] is not None


def test_missing_checkpoint_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        Predictor.from_checkpoint(tmp_path / "nope.pt")


def test_trainer_restores_best_weights(tmp_path):
    """Regression: test evaluation used the LAST epoch's weights, not the best."""
    torch.manual_seed(0)
    X, y = np.random.randn(40, 8).astype("float32"), np.random.randn(40).astype("float32")
    model = MolecularMLP(input_dim=8, hidden_dims=[16], use_batch_norm=False)
    loader = DataLoader(MolecularDataset(X[:30], y[:30]), batch_size=10)
    val = DataLoader(MolecularDataset(X[30:], y[30:]), batch_size=10)
    t = Trainer(model, loader, val, epochs=15, patience=100, checkpoint_dir=str(tmp_path),
                device="cpu", learning_rate=0.05)
    t.fit()
    loss_now, _ = t.evaluate(val)
    assert abs(loss_now - t.best_val_loss) < 1e-5
