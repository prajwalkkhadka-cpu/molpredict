# MolPredict

Predict molecular **LogP** (lipophilicity) from a SMILES string.
Pipeline: SMILES -> RDKit features (Morgan fingerprint + 8 descriptors) -> PyTorch MLP.

> **Status: prototype.** The code is a working, tested end-to-end pipeline, but the bundled
> dataset has only 50 molecules, so the numbers below are illustrative, not a benchmark.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .                      # optional: installs `molpredict-train` / `molpredict-predict`

python scripts/train.py               # trains and writes checkpoints/best_model.pt + metrics.json
python predict.py --name aspirin
python predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
python demo.py                        # interactive menu
python -m pytest -q                   # 27 tests
```

Without a trained checkpoint, `predict.py` and `demo.py` say so and show RDKit-computed values
only (labelled "RDKit Crippen estimate"). They never fake a model prediction. Invalid SMILES are rejected.

## What each part does

| Path | Purpose |
|---|---|
| `molpredict/data/preprocessing.py` | load, clean (invalid/duplicate/unicode-minus handling), **scaffold** or random split, target scaling |
| `molpredict/features/engineering.py` | Morgan fingerprints + descriptors; scaler fitted on train only; state is saved for inference |
| `molpredict/models/mlp.py` | feed-forward regressor (Linear, BatchNorm, activation, Dropout) |
| `molpredict/training/trainer.py` | AdamW, warm-up + cosine LR, grad clipping, early stopping, restores best weights |
| `molpredict/pipeline.py` | full training run, plus baselines evaluated on the same test molecules |
| `molpredict/inference.py` | `Predictor.from_checkpoint(...)`: reproduces training featurisation and target scaling |
| `molpredict/chem.py` | RDKit properties and Lipinski rule of five (real values) |
| `config/config.yaml` | every key is read by the pipeline |

## Example run on the bundled data

Scaffold split, 36 train / 6 val / 7 test molecules, seed 42:

| Model (test set, n=7) | RMSE | R^2 |
|---|---|---|
| Predict train mean | 1.84 | -0.30 |
| Ridge on same features | 1.40 | 0.25 |
| **MLP (this repo)** | **1.03** | **0.60** |
| RDKit Crippen LogP (no training) | 0.88 | 0.70 |

With 7 test molecules these differences are well within noise (validation R^2 was negative).
RDKit's own Crippen estimate is the baseline to beat; to get a meaningful result, train on a real
dataset such as MoleculeNet ESOL or Lipophilicity (set `data.path` / `target_col` in the config).

## Known limitations

* **Dataset:** `data/sample_data.csv` is tiny. Its `logP` labels are unverified, and a few look
  questionable (e.g. salicylic acid is listed at 1.19, the same as aspirin). Verify against a
  reference source before drawing conclusions. `mol_weight` and `tpsa` columns were recomputed with
  RDKit (the originals disagreed with the SMILES); the pipeline does not use them.
* The demo's "reference" LogP for preset drugs comes from the same small set the model trains on,
  so those errors are not a fair test.
* Crippen `MolLogP` is intentionally **not** an input feature (it would leak the target).
* Single-task regression only; no GNN is implemented.
