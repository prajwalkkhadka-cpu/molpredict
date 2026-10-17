# Changes from the original upload

## Honesty / correctness
- `demo.py` / `predict.py` no longer fake inference (it counted C/O/N characters and added sin() noise).
  They now use RDKit for properties and the trained checkpoint for LogP; invalid SMILES are rejected.
- README rewritten: removed hand-written sample output and "production-grade" claims; added real results + limitations.
- Lipinski check uses real RDKit values instead of guessed MW/HBD/HBA.

## Crashes / broken packaging
- `import molpredict` crashed (imported non-existent `molpredict.models.gnn`); removed GNN claims and dead graph dataset.
- `python scripts/train.py` could not import the package from a fresh clone; fixed.
- Subpackages had no `__init__.py` (silently dropped by `find_packages`); added.
- `setup.py` entry points referenced non-existent scripts; now point to real functions. Added `pyproject.toml`.
- Random split crashed on small data (decile stratification); now degrades safely.

## Data bugs
- 8 rows with a unicode minus (U+2212) were silently turned into NaN and dropped; now parsed (and the file normalised).
- `mol_weight` / `tpsa` columns contradicted the SMILES; recomputed with RDKit.

## ML-pipeline bugs
- Test set was evaluated with the LAST-epoch weights, not the best; trainer now restores best weights.
- Checkpoint lacked feature/target scalers, so inference was impossible; they are now saved.
- Default split is now a scaffold split (random splits inflate molecular-ML metrics).
- Removed `MolLogP` from input descriptors (target leakage); added baselines (mean, Ridge, RDKit Crippen).
- Config keys that nothing read (`pca`, `use_chirality`, `gnn`, ...) removed; `gradient_clip`, `min_delta`,
  `warmup_epochs`, `early_stopping.enabled`, `device` are now actually wired up.
- Deprecated RDKit fingerprint API replaced; unfitted descriptor scaler now raises instead of silently skipping scaling.
- Metrics return NaN instead of a misleading 0.0 for undefined correlations.
- Final batch of size 1 would crash BatchNorm training; handled via `drop_last`.
- Reproducible seeding added.

## Tests
27 tests (was 4), including a regression test for each bug above and a train -> save -> load -> predict roundtrip.
