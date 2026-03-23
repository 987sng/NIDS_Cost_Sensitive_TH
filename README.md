# Cost-Sensitive NIDS via Per-Class Threshold Optimization

Network Intrusion Detection System (NIDS) that reduces operational cost by applying cost-sensitive, per-class decision thresholds optimized on a validation set.

## Overview

Standard classifiers use a fixed threshold (τ = 0.5), which does not account for the asymmetric cost of missed attacks (FN) vs. false alarms (FP). This work:

1. **Global threshold optimization** – finds a single τ* that minimizes `cost = FN × C_FN + FP × C_FP`
2. **Per-class threshold optimization (Algorithm 1)** – learns an independent τ for each attack class, then uses a union rule: flag as attack if *any* class exceeds its threshold

Primary cost setting: C_FN = 10, C_FP = 1 (Elkan 2001 / KDD Cup 99)

## Datasets

| Dataset | Split | Classes |
|---------|-------|---------|
| [CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | 60/20/20 (train/val/test) | 15 |
| [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 60/20/20 | 10 |

Place raw files at:
```
./data/cicids2018/          # CSV files from CIC-IDS2018
../실험데이터/unsw/UNSW_NB15_training-set.csv
../실험데이터/unsw/UNSW_NB15_testing-set.csv
```

## Models

- Naive Bayes (GaussianNB)
- Decision Tree (`max_depth=10`)
- Random Forest (`n_estimators=100, max_depth=20`)
- XGBoost (`n_estimators=500, lr=0.1, max_depth=10`)

## Usage

```bash
# Full run (5 seeds)
python main.py

# Quick run (seed=42 only)
python main.py --quick
```

Processed data and per-seed results are cached under `processed_data_*.pkl` and `checkpoints/`. Re-runs resume automatically.

**Outputs:**
- `experimental_results.txt` — Tables 1–5
- `fig/` — Figures 1–5 (cost curves, bar chart, confusion matrix, feature importance, cost sensitivity)

## Requirements

```
numpy pandas polars scikit-learn xgboost joblib matplotlib
```

## Results Summary

Running the experiment produces the following outputs:

**CIC-IDS2018**
- 4-algorithm benchmark (NB / DT / RF / XGBoost) with fixed vs. optimized threshold costs and recall
- Comparison table: RF baseline vs. XGBoost global/per-class threshold optimization
- Per-class cost breakdown with optimized threshold per attack type

**UNSW-NB15**
- Same 4-algorithm benchmark for generalizability validation
- Per-class cost breakdown with optimized threshold per attack type

**Cross-dataset**
- Cost ratio sensitivity analysis (FN:FP = 5:1 / 10:1 / 20:1)

See `experimental_results.txt` for full numerical results after running.
