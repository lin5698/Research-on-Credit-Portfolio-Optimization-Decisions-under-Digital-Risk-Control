# Credit Portfolio Optimization Framework

This repository contains code, validation data, and benchmark outputs for a three-stage credit portfolio optimization pipeline.

## Repository contents

- `data/`
  Replication dataset used by the top-level implementation.
- `Paper_Implementation_Code/`
  A compact implementation snapshot kept with its own `README.md`.
- `stage1_prediction.py`, `stage2_evaluation.py`, `stage3_optimization.py`
  The three-stage pipeline for interpretable prediction, enterprise evaluation, and portfolio optimization.
- `supplementary_experiments.py`
  Repeated-run benchmark script used to generate matched-budget benchmark comparisons.
- `output/experiments/reviewer_supplement/`
  Benchmark outputs, including summary tables, per-run results, and statistical tests.

## Current benchmark scope

The current repository state includes:

- a genuinely budget-matched supplementary experiment for `SA-NA`, `SA-only`, `NSGA-II`, `MOEA/D`, and `SPEA2`;
- summary tables and statistical tests under `output/experiments/reviewer_supplement/`;
- a compact implementation snapshot under `Paper_Implementation_Code/`.

## Main files for readers

- Benchmark summary:
  `output/experiments/reviewer_supplement/summary.md`
- Top-level entry point:
  `main.py`

## Running the code

The top-level scripts were developed against Python 3.8+ and standard scientific Python libraries, including:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `shap`
- `matplotlib`

Typical entry points:

```bash
python main.py
python supplementary_experiments.py
```

## Notes

- The optimization benchmark outputs in `output/experiments/reviewer_supplement/` are generated artifacts and are intentionally versioned here so the reported results remain reproducible.
- The repository includes both the working project root and the compact `Paper_Implementation_Code/` subfolder because they were used for different packaging and replication needs during manuscript preparation.
