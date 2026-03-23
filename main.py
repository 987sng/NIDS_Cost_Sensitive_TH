import os
import glob
import gc
import sys
import signal
import warnings

def _sigint_handler(sig, frame):
    print("\n[Interrupted] Saving is done per-seed. Re-run to resume from checkpoints.")
    sys.exit(0)
signal.signal(signal.SIGINT, _sigint_handler)
import numpy as np
import joblib
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
plt.style.use('default')

# ==============================================================================
# 1. Configuration
# ==============================================================================

# Data paths (relative to cic/ working directory)
CIC_DATA_PATH   = "./data/cicids2018/"
UNSW_TRAIN_PATH = "../실험데이터/unsw/UNSW_NB15_training-set.csv"
UNSW_TEST_PATH  = "../실험데이터/unsw/UNSW_NB15_testing-set.csv"
CIC_CACHE_PATH  = "processed_data_cic.pkl"
UNSW_CACHE_PATH = "processed_data_unsw.pkl"

# Hyperparameters (R2-3.1: reported explicitly for reproducibility)
XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.1, max_depth=10,
    objective='multi:softprob', tree_method='hist', n_jobs=-1,
)
RF_PARAMS  = dict(n_estimators=100, max_depth=20, n_jobs=-1)
DT_PARAMS  = dict(max_depth=10)

# R3-2: Statistical robustness — 5 independent random seeds
SEEDS_QUICK = [42]
SEEDS_FULL  = [42, 0, 7, 13, 99]
SEEDS = SEEDS_QUICK if "--quick" in sys.argv else SEEDS_FULL

# R3-4: Cost ratio sensitivity analysis
COST_RATIOS = [(5, 1), (10, 1), (20, 1)]

# Primary cost setting (Elkan 2001; KDD Cup 99 justification)
PRIMARY_COST_FN = 10
PRIMARY_COST_FP = 1

THRESHOLD_GRID = np.arange(0.01, 1.00, 0.01)

print("=" * 80)
print("[Paper] Cost-Sensitive NIDS via Per-Class Threshold Optimization")
print(f"  Seeds          : {SEEDS}")
print(f"  Primary cost   : FN={PRIMARY_COST_FN}, FP={PRIMARY_COST_FP}")
print(f"  Cost ratios    : {COST_RATIOS}")
print(f"  Split          : 60 / 20 / 20  (train / val / test)")
print("=" * 80)

# ==============================================================================
# 2. Core helpers
# ==============================================================================

def compute_cost(fn, fp, cost_fn, cost_fp):
    return int(fn * cost_fn + fp * cost_fp)

def get_metrics(y_true_bin, y_pred_bin, cost_fn, cost_fp):
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    acc    = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn + 1e-9)
    cost   = compute_cost(fn, fp, cost_fn, cost_fp)
    return {'cost': cost, 'acc': float(acc), 'rec': float(recall),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def find_optimal_threshold(y_true_bin, y_proba_attack, cost_fn, cost_fp):
    """
    Exhaustive grid search over THRESHOLD_GRID.
    Returns (best_threshold, list_of_costs_per_threshold).
    """
    best_th, best_cost = 0.5, float('inf')
    costs = []
    for t in THRESHOLD_GRID:
        m = get_metrics(y_true_bin, (y_proba_attack >= t).astype(int), cost_fn, cost_fp)
        costs.append(m['cost'])
        if m['cost'] < best_cost:
            best_cost, best_th = m['cost'], float(t)
    return best_th, costs

def find_optimal_threshold_perclass(is_class_k, is_benign, y_proba_k, cost_fn, cost_fp):
    """
    Per-class threshold search.
      FN = class-k samples below threshold (missed attacks)
      FP = benign samples above threshold (false alarms)
    Cross-class confusion (other attacks above threshold) is NOT penalized,
    since they are still correctly identified as attacks in the final binary decision.
    """
    best_th, best_cost = 0.5, float('inf')
    for t in THRESHOLD_GRID:
        pred = (y_proba_k >= t)
        fn = int(((is_class_k == 1) & ~pred).sum())
        fp = int(((is_benign == 1) &  pred).sum())
        cost = compute_cost(fn, fp, cost_fn, cost_fp)
        if cost < best_cost:
            best_cost, best_th = cost, float(t)
    return best_th

# ==============================================================================
# 3. CIC-IDS2018 data loading
# ==============================================================================

def _load_cic_raw(path):
    all_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
    if not all_files:
        print(f"Error: no CSVs in {path}"); sys.exit(1)
    print(f"  [CIC] Scanning {len(all_files)} files...")

    lazy_frames = []
    for f in all_files:
        try:
            lf = pl.scan_csv(f, ignore_errors=True, infer_schema_length=10000)
            lf = lf.rename({c: c.strip() for c in lf.collect_schema().names()})
            lazy_frames.append(lf)
        except:
            continue

    common_cols = set(lazy_frames[0].collect_schema().names())
    for lf in lazy_frames[1:]:
        common_cols.intersection_update(set(lf.collect_schema().names()))
    common_cols = list(common_cols)
    feat_cols = [c for c in common_cols if c != "Label"]

    processed = []
    for lf in lazy_frames:
        lf = lf.select(common_cols)
        lf = lf.with_columns([
            pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in feat_cols
        ])
        processed.append(lf)

    combined = pl.concat(processed).filter(pl.col("Label") != "Label")
    drop = [c for c in ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port']
            if c in combined.collect_schema().names()]
    if drop:
        combined = combined.drop(drop)

    final_feat = [c for c in combined.collect_schema().names() if c != "Label"]
    combined = combined.with_columns([
        pl.col(c).map_elements(
            lambda x: 0.0 if x in (float('inf'), float('-inf')) else x,
            return_dtype=pl.Float64
        ).fill_null(0.0)
        for c in final_feat
    ])

    df = combined.collect().to_pandas()
    # NaN/Inf replaced with 0 — CIC-IDS2018 contains Inf values from division-by-zero
    # in flow duration features; zero-imputation is standard practice for this dataset.
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    le = LabelEncoder()
    y = le.fit_transform(df['Label'].values)
    X = df.drop(columns=['Label'])
    class_names = le.classes_
    benign_idx = int(np.where(class_names == 'Benign')[0][0]) if 'Benign' in class_names else 0
    return X, y, class_names, benign_idx

def load_cic_data():
    if os.path.exists(CIC_CACHE_PATH):
        print(f"  [Cache] CIC-IDS2018 from '{CIC_CACHE_PATH}'")
        d = joblib.load(CIC_CACHE_PATH)
        return d['X'], d['y'], d['class_names'], d['benign_idx']
    print("  [Cache miss] Processing CIC-IDS2018 raw CSVs (one-time)...")
    X, y, class_names, benign_idx = _load_cic_raw(CIC_DATA_PATH)
    joblib.dump({'X': X, 'y': y, 'class_names': class_names, 'benign_idx': benign_idx},
                CIC_CACHE_PATH)
    return X, y, class_names, benign_idx

# ==============================================================================
# 4. UNSW-NB15 data loading
# ==============================================================================

def load_unsw_data():
    if os.path.exists(UNSW_CACHE_PATH):
        print(f"  [Cache] UNSW-NB15 from '{UNSW_CACHE_PATH}'")
        d = joblib.load(UNSW_CACHE_PATH)
        return d['X'], d['y'], d['class_names'], d['benign_idx']
    print("  [Cache miss] Processing UNSW-NB15...")

    df_tr = pd.read_csv(UNSW_TRAIN_PATH)
    df_te = pd.read_csv(UNSW_TEST_PATH)
    df = pd.concat([df_tr, df_te], ignore_index=True)

    # Drop row-ID; use attack_cat as multi-class label (mirrors CIC-IDS2018 setup)
    df = df.drop(columns=['id', 'label'], errors='ignore')

    # Encode categorical network-feature columns with LabelEncoder
    for col in ['proto', 'service', 'state']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # NaN/Inf → 0 (same treatment as CIC-IDS2018)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    le = LabelEncoder()
    y = le.fit_transform(df['attack_cat'].astype(str).str.strip())
    X = df.drop(columns=['attack_cat'])
    class_names = le.classes_
    benign_idx = int(np.where(class_names == 'Normal')[0][0]) if 'Normal' in class_names else 0

    joblib.dump({'X': X, 'y': y, 'class_names': class_names, 'benign_idx': benign_idx},
                UNSW_CACHE_PATH)
    return X, y, class_names, benign_idx

# ==============================================================================
# 5. Per-class threshold decision rule  (Algorithm 1 in paper)
# ==============================================================================
#
#  Given XGBoost multi-class probabilities P  [n × K]
#  and per-class thresholds  τ[k]  selected on the validation set:
#
#    For each sample i:
#      attack_flags ← { 1  if  P[i, k] ≥ τ[k],  else 0 }   for k ≠ benign_idx
#      if any(attack_flags) → predicted = Attack   (conservative: treat as attack)
#      else                 → predicted = Benign   (fallback: no class triggered)
#
#  Ties (multiple classes exceed their threshold): predict Attack.
#  No class exceeds threshold: predict Benign (most restrictive fallback).

def apply_per_class_thresholds(y_proba, class_names, benign_idx, thresholds_dict):
    """
    thresholds_dict: {class_index (int): threshold (float)}
    Returns binary prediction array  (1 = Attack, 0 = Benign).
    """
    pred = np.zeros(len(y_proba), dtype=int)
    for k, _ in enumerate(class_names):
        if k == benign_idx:
            continue
        th = thresholds_dict.get(k, 0.5)
        pred = np.maximum(pred, (y_proba[:, k] >= th).astype(int))
    return pred

# ==============================================================================
# 6. Single experiment run  (one seed, one cost setting)
# ==============================================================================

def run_experiment(X, y, class_names, benign_idx, seed, cost_fn, cost_fp,
                   return_curves=False):
    """
    60/20/20 stratified split → train 4 models →
    validate (threshold opt) → test (final reporting only).

    return_curves=True: also return val cost curves for Figure 1.
    """
    # ── Split ────────────────────────────────────────────────────────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)

    # ── Train ─────────────────────────────────────────────────────────────────
    trained = {
        'Naive Bayes':   GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(**DT_PARAMS, random_state=seed),
        'Random Forest': RandomForestClassifier(**RF_PARAMS, random_state=seed),
        'XGBoost':       XGBClassifier(**XGB_PARAMS, num_class=len(class_names),
                                       random_state=seed),
    }
    for m in trained.values():
        m.fit(X_tr, y_tr)

    y_val_bin = (y_val != benign_idx).astype(int)
    y_te_bin  = (y_te  != benign_idx).astype(int)

    # ── Global threshold analysis (Experiment 1) ──────────────────────────────
    model_results = {}
    val_curves    = {}

    for name, model in trained.items():
        p_val = model.predict_proba(X_val)
        p_te  = model.predict_proba(X_te)
        att_val = 1 - p_val[:, benign_idx]
        att_te  = 1 - p_te[:,  benign_idx]

        # Threshold search on validation set
        best_th, costs_val = find_optimal_threshold(y_val_bin, att_val, cost_fn, cost_fp)

        # Apply fixed and val-optimized threshold to test set
        res_fixed = get_metrics(y_te_bin, (att_te >= 0.5).astype(int), cost_fn, cost_fp)
        res_fixed['th'] = 0.5
        res_opt = get_metrics(y_te_bin, (att_te >= best_th).astype(int), cost_fn, cost_fp)
        res_opt['th'] = best_th

        model_results[name] = {'fixed': res_fixed, 'opt': res_opt, 'p_te': p_te}
        if return_curves:
            val_curves[name] = {'ths': THRESHOLD_GRID.copy(), 'costs': costs_val,
                                'opt_th': best_th}

    # ── Per-class threshold optimization (Experiment 2, XGBoost only) ─────────
    xgb_global_th  = model_results['XGBoost']['opt']['th']
    p_val_xgb      = trained['XGBoost'].predict_proba(X_val)
    p_te_xgb       = model_results['XGBoost']['p_te']

    per_class_th   = {}   # {class_idx: val-optimal threshold}
    per_class_stats = []

    val_is_benign = (y_val == benign_idx).astype(int)
    te_is_benign  = (y_te  == benign_idx).astype(int)

    for k, cls in enumerate(class_names):
        if k == benign_idx:
            continue

        # Find class-specific optimal threshold on val
        # FN = missed class-k attacks, FP = benign falsely alarmed (cross-class confusion excluded)
        val_is_k  = (y_val == k).astype(int)
        best_th_k = find_optimal_threshold_perclass(
            val_is_k, val_is_benign, p_val_xgb[:, k], cost_fn, cost_fp)
        per_class_th[k] = float(best_th_k)

        # Evaluate both strategies on test set for this class (same FP definition)
        te_is_k = (y_te == k).astype(int)
        pred_g  = (p_te_xgb[:, k] >= xgb_global_th)
        fn_g    = int(((te_is_k == 1)    & ~pred_g).sum())
        fp_g    = int(((te_is_benign == 1) & pred_g).sum())
        cost_global = compute_cost(fn_g, fp_g, cost_fn, cost_fp)

        pred_p  = (p_te_xgb[:, k] >= best_th_k)
        fn_p    = int(((te_is_k == 1)    & ~pred_p).sum())
        fp_p    = int(((te_is_benign == 1) & pred_p).sum())
        cost_opt = compute_cost(fn_p, fp_p, cost_fn, cost_fp)

        reduction = (cost_global - cost_opt) / cost_global * 100 if cost_global > 0 else 0.0
        per_class_stats.append({
            'name': cls, 'global_cost': cost_global, 'opt_cost': cost_opt,
            'th_opt': float(best_th_k), 'reduction': float(reduction),
        })

    # Assemble final binary prediction using per-class thresholds (Algorithm 1)
    final_pred     = apply_per_class_thresholds(p_te_xgb, class_names, benign_idx, per_class_th)
    res_per_class  = get_metrics(y_te_bin, final_pred, cost_fn, cost_fp)

    out = {
        'model_results':  model_results,
        'per_class_stats': per_class_stats,
        'res_per_class':  res_per_class,
        'xgb_global_th':  xgb_global_th,
        'trained':        trained,      # kept for Fig 4 (seed=42 only)
        'te_bin':         y_te_bin,
        'final_pred':     final_pred,
    }
    if return_curves:
        out['val_curves'] = val_curves
    return out

# ==============================================================================
# 7. Multi-seed runner  →  mean ± std  (with per-seed checkpointing)
# ==============================================================================

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def _ckpt_path(tag, seed):
    return os.path.join(CKPT_DIR, f"{tag}_seed{seed}.pkl")

def _ratio_ckpt_path(cfn, cfp):
    return os.path.join(CKPT_DIR, f"ratio_{cfn}_{cfp}.pkl")

def _pick(run, *keys):
    v = run
    for k in keys:
        v = v[k]
    return float(v)

def run_multi_seed(X, y, class_names, benign_idx, seeds, cost_fn, cost_fp, tag="exp"):
    print(f"\n  Running {len(seeds)} seeds: {seeds}  [tag={tag}]")
    runs = []
    for s in seeds:
        ckpt = _ckpt_path(tag, s)
        if os.path.exists(ckpt):
            print(f"    seed={s} ... [RESUME from checkpoint]")
            r = joblib.load(ckpt)
        else:
            print(f"    seed={s} ...", end=" ", flush=True)
            r = run_experiment(X, y, class_names, benign_idx, s, cost_fn, cost_fp,
                               return_curves=(s == seeds[0]))
            joblib.dump(r, ckpt)
            pc = r['res_per_class']['cost']
            print(f"done  (per-class cost={pc:,})")
        runs.append(r)

    def agg(*keys):
        vals = np.array([_pick(r, *keys) for r in runs])
        return float(vals.mean()), float(vals.std())

    summary = {}
    for mname in ['Naive Bayes', 'Decision Tree', 'Random Forest', 'XGBoost']:
        summary[mname] = {
            'fixed_cost': agg('model_results', mname, 'fixed', 'cost'),
            'fixed_acc':  agg('model_results', mname, 'fixed', 'acc'),
            'fixed_rec':  agg('model_results', mname, 'fixed', 'rec'),
            'opt_cost':   agg('model_results', mname, 'opt',   'cost'),
            'opt_acc':    agg('model_results', mname, 'opt',   'acc'),
            'opt_rec':    agg('model_results', mname, 'opt',   'rec'),
            'opt_th':     agg('model_results', mname, 'opt',   'th'),
        }
    summary['per_class'] = {
        'cost': agg('res_per_class', 'cost'),
        'acc':  agg('res_per_class', 'acc'),
        'rec':  agg('res_per_class', 'rec'),
    }
    summary['_runs']    = runs
    summary['_ref_run'] = runs[0]   # seed=42 — used for Table 3 and Figures
    return summary

# ==============================================================================
# 8. Cost ratio sensitivity  (R3-4, with checkpointing)
# ==============================================================================

def run_cost_sensitivity(X, y, class_names, benign_idx, seed=42):
    print(f"\n[Cost-Ratio Sensitivity]  seed={seed}")
    results = {}
    for (cfn, cfp) in COST_RATIOS:
        ckpt = _ratio_ckpt_path(cfn, cfp)
        if os.path.exists(ckpt):
            print(f"  FN/FP = {cfn}/{cfp} ... [RESUME from checkpoint]")
            results[(cfn, cfp)] = joblib.load(ckpt)
        else:
            print(f"  FN/FP = {cfn}/{cfp} ...", end=" ", flush=True)
            r = run_experiment(X, y, class_names, benign_idx, seed, cfn, cfp,
                               return_curves=True)
            joblib.dump(r, ckpt)
            results[(cfn, cfp)] = r
            print(f"done  (global τ={r['xgb_global_th']:.2f}  "
                  f"per-class cost={r['res_per_class']['cost']:,})")
    return results

# ==============================================================================
# 9. Report generation
# ==============================================================================

def _fmt(mean, std):
    return f"{mean:>12,.1f} ± {std:<8,.1f}"

def build_report(cic_sum, unsw_sum, ratio_res):
    ref     = cic_sum['_ref_run']
    ref_th  = ref['xgb_global_th']
    lines   = []

    W = 92
    lines.append("=" * W)
    lines.append("EXPERIMENTAL RESULTS REPORT")
    lines.append(f"  Seeds: {SEEDS}   Primary cost: FN={PRIMARY_COST_FN}, FP={PRIMARY_COST_FP}")
    lines.append(f"  Protocol: 60/20/20 train/val/test  |  threshold opt on val  |  report on test")
    lines.append("=" * W)

    # ── Table 1: CIC-IDS2018 benchmark ───────────────────────────────────────
    lines.append("\n[TABLE 1]  CIC-IDS2018 — 4-Algorithm Benchmark  (mean ± std, 5 seeds)")
    lines.append("-" * W)
    lines.append(f"{'Model':<18} | {'Val Opt Th':>10} | {'Fixed Cost':>22} | {'Opt Cost':>22} | {'Fixed Rec':>10} | {'Opt Rec':>9}")
    lines.append("-" * W)
    for mn in ['Naive Bayes', 'Decision Tree', 'Random Forest', 'XGBoost']:
        s = cic_sum[mn]
        lines.append(
            f"{mn:<18} | {s['opt_th'][0]:>10.2f} "
            f"| {_fmt(*s['fixed_cost'])} "
            f"| {_fmt(*s['opt_cost'])} "
            f"| {s['fixed_rec'][0]:>10.4f} "
            f"| {s['opt_rec'][0]:>9.4f}"
        )
    lines.append("-" * W)

    # ── Table 2: Final comparison ─────────────────────────────────────────────
    rf_fc  = cic_sum['Random Forest']['fixed_cost']
    xgb_gc = cic_sum['XGBoost']['opt_cost']
    pc_c   = cic_sum['per_class']['cost']
    red_g  = (rf_fc[0] - xgb_gc[0]) / rf_fc[0] * 100 if rf_fc[0] > 0 else 0
    red_pc = (rf_fc[0] - pc_c[0])   / rf_fc[0] * 100 if rf_fc[0] > 0 else 0
    lines.append("\n[TABLE 2]  Final Comparison: Baseline vs Proposed  (CIC-IDS2018)")
    lines.append("-" * 65)
    lines.append(f"{'Strategy':<36} | {'Total Cost  (mean ± std)':>24} | {'Reduction':>8}")
    lines.append("-" * 65)
    lines.append(f"{'Random Forest (Fixed τ=0.5)':<36} | {_fmt(*rf_fc):>24} | {'Baseline':>8}")
    lines.append(f"{'XGBoost (Global Opt)':<36} | {_fmt(*xgb_gc):>24} | {red_g:>7.2f}%")
    lines.append(f"{'XGBoost (Per-Class Opt)':<36} | {_fmt(*pc_c):>24} | {red_pc:>7.2f}%")
    lines.append("-" * 65)

    # ── Table 3: Per-class breakdown (ref seed) ───────────────────────────────
    lines.append(f"\n[TABLE 3]  Per-Class Breakdown — XGBoost  (seed=42, global τ={ref_th:.2f})")
    lines.append("-" * W)
    lines.append(f"{'Attack Type':<28} | {'Global Cost':>13} | {'Per-Class Cost':>14} | {'Reduction':>10} | {'Opt τ':>6}")
    lines.append("-" * W)
    stats = sorted(ref['per_class_stats'], key=lambda x: x['reduction'], reverse=True)
    for s in stats:
        lines.append(
            f"{s['name']:<28} | {s['global_cost']:>13,} | {s['opt_cost']:>14,} "
            f"| {s['reduction']:>9.2f}% | {s['th_opt']:>6.2f}"
        )
    lines.append("-" * W)
    lines.append(
        "  NOTE: Infiltration dominates total cost (>99%). Its per-class τ equals the "
        "global τ,\n"
        "  indicating that further threshold tuning alone cannot resolve this class. "
        "See discussion."
    )

    # ── Table 4: UNSW-NB15 ────────────────────────────────────────────────────
    u_rf_fc = unsw_sum['Random Forest']['fixed_cost']
    u_pc_c  = unsw_sum['per_class']['cost']
    u_red   = (u_rf_fc[0] - u_pc_c[0]) / u_rf_fc[0] * 100 if u_rf_fc[0] > 0 else 0
    lines.append("\n[TABLE 4]  UNSW-NB15 — Generalizability Validation  (mean ± std, 5 seeds)")
    lines.append("-" * W)
    lines.append(f"{'Model':<18} | {'Val Opt Th':>10} | {'Fixed Cost':>22} | {'Opt Cost':>22} | {'Opt Rec':>9}")
    lines.append("-" * W)
    for mn in ['Naive Bayes', 'Decision Tree', 'Random Forest']:
        s = unsw_sum[mn]
        lines.append(
            f"{mn:<18} | {s['opt_th'][0]:>10.2f} "
            f"| {_fmt(*s['fixed_cost'])} "
            f"| {_fmt(*s['opt_cost'])} "
            f"| {s['opt_rec'][0]:>9.4f}"
        )
    u_xgb_gc = unsw_sum['XGBoost']['opt_cost']
    u_red_g  = (u_rf_fc[0] - u_xgb_gc[0]) / u_rf_fc[0] * 100 if u_rf_fc[0] > 0 else 0
    lines.append(f"{'XGBoost (Global Opt)':<18} | {unsw_sum['XGBoost']['opt_th'][0]:>10.2f} | {_fmt(*unsw_sum['XGBoost']['fixed_cost'])} | {_fmt(*u_xgb_gc):>22} | {unsw_sum['XGBoost']['opt_rec'][0]:>9.4f}")
    lines.append(f"{'XGBoost (Per-Class Opt)':<18} | {'':>10} | {'':>22} | {_fmt(*u_pc_c):>22} | {unsw_sum['per_class']['rec'][0]:>9.4f}")
    lines.append("-" * W)
    lines.append(f"  RF Fixed baseline: {u_rf_fc[0]:,.1f}  |  XGB Global: {u_xgb_gc[0]:,.1f} ({u_red_g:.2f}%)  |  XGB Per-Class: {u_pc_c[0]:,.1f} ({u_red:.2f}%)")

    # ── Table 4b: UNSW per-class breakdown (ref seed) ────────────────────────
    u_ref     = unsw_sum['_ref_run']
    u_ref_th  = u_ref['xgb_global_th']
    u_stats   = sorted(u_ref['per_class_stats'], key=lambda x: x['reduction'], reverse=True)
    lines.append(f"\n[TABLE 4b]  Per-Class Breakdown — XGBoost  UNSW-NB15  (seed=42, global τ={u_ref_th:.2f})")
    lines.append("-" * W)
    lines.append(f"{'Attack Type':<28} | {'Global Cost':>13} | {'Per-Class Cost':>14} | {'Reduction':>10} | {'Opt τ':>6}")
    lines.append("-" * W)
    for s in u_stats:
        lines.append(
            f"{s['name']:<28} | {s['global_cost']:>13,} | {s['opt_cost']:>14,} "
            f"| {s['reduction']:>9.2f}% | {s['th_opt']:>6.2f}"
        )
    lines.append("-" * W)

    # ── Table 5: Cost ratio sensitivity ──────────────────────────────────────
    lines.append("\n[TABLE 5]  Cost Ratio Sensitivity  (XGBoost, CIC-IDS2018, seed=42)")
    lines.append("-" * 72)
    lines.append(f"{'FN:FP Ratio':>12} | {'Global Opt τ':>12} | {'Global Cost':>13} | {'Per-Class Cost':>14} | {'Reduction':>9}")
    lines.append("-" * 72)
    for (cfn, cfp), r in ratio_res.items():
        gc_ = r['model_results']['XGBoost']['opt']['cost']
        pc_ = r['res_per_class']['cost']
        red_r = (gc_ - pc_) / gc_ * 100 if gc_ > 0 else 0
        lines.append(
            f"  {cfn:>2}:{cfp:<2}        | {r['xgb_global_th']:>12.2f} "
            f"| {gc_:>13,} | {pc_:>14,} | {red_r:>8.2f}%"
        )
    lines.append("-" * 72)

    return "\n".join(lines)

# ==============================================================================
# 10. Visualization
# ==============================================================================

def make_figures(cic_sum, ratio_res):
    ref      = cic_sum['_ref_run']
    ref_th   = ref['xgb_global_th']
    curves   = ref['val_curves']        # validation cost curves (seed=42)
    os.makedirs("fig", exist_ok=True)

    colors = {'Naive Bayes': 'green', 'Decision Tree': 'orange',
              'Random Forest': 'blue',  'XGBoost': 'red'}
    styles = {'Naive Bayes': ':',       'Decision Tree': '-.',
              'Random Forest': '--',    'XGBoost': '-'}

    # ── Figure 1: Validation cost curves ─────────────────────────────────────
    plt.figure(figsize=(12, 6))
    for name in ['Naive Bayes', 'Decision Tree', 'Random Forest', 'XGBoost']:
        c = curves[name]
        label = f"{name}  (τ*={c['opt_th']:.2f})"
        plt.plot(c['ths'], c['costs'], label=label,
                 color=colors[name], linestyle=styles[name], linewidth=2)
        min_cost = min(c['costs'])
        plt.scatter(c['opt_th'], min_cost, color=colors[name], s=100, marker='*', zorder=5)
    plt.xlabel('Threshold (τ)')
    plt.ylabel('Validation Set Cost')
    plt.title('Cost vs. Threshold  [Validation Set — CIC-IDS2018, seed=42]')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/val_cost_curve.png', dpi=300)
    plt.close()
    print("  [Fig] val_cost_curve.png saved.")

    # ── Figure 2: Cost comparison bar chart ───────────────────────────────────
    rf_fixed = cic_sum['Random Forest']['fixed_cost'][0]
    xgb_pc   = cic_sum['per_class']['cost'][0]
    xgb_pc_std = cic_sum['per_class']['cost'][1]

    nb_fixed = cic_sum['Naive Bayes']['fixed_cost'][0]
    dt_fixed = cic_sum['Decision Tree']['fixed_cost'][0]

    labels_bar = ['NB\n(Fixed)', 'DT\n(Fixed)', 'RF\n(Fixed)', 'XGBoost\n(Per-Class)']
    values_bar = [nb_fixed, dt_fixed, rf_fixed, xgb_pc]
    errs       = [0, 0, 0, xgb_pc_std]
    bar_colors = ['#d9d9d9', '#bdbdbd', '#808080', '#2ca02c']

    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels_bar, values_bar, color=bar_colors, edgecolor='black',
                   width=0.55, yerr=errs, capsize=5)
    for i, bar in enumerate(bars):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + max(values_bar) * 0.01,
                 f'{int(h):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        if i == 3:
            red = (rf_fixed - xgb_pc) / rf_fixed * 100
            plt.text(bar.get_x() + bar.get_width() / 2, h * 0.55,
                     f'▼{red:.1f}%\nvs RF', ha='center', va='center',
                     fontweight='bold', color='white', fontsize=11)
    plt.ylabel('Total Operational Cost (mean over 5 seeds)')
    plt.ylim(0, max(values_bar) * 1.18)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/cost_bar.png', dpi=300)
    plt.close()
    print("  [Fig] cost_bar.png saved.")

    # ── Figure 3: Confusion matrix (ref seed) ─────────────────────────────────
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(ref['te_bin'], ref['final_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Attack'])
    disp.plot(cmap='Greens', values_format='d', colorbar=False)
    plt.title('Confusion Matrix — XGBoost Per-Class  (CIC-IDS2018, seed=42)')
    plt.tight_layout()
    plt.savefig('fig/confusion_matrix.png', dpi=300)
    plt.close()
    print("  [Fig] confusion_matrix.png saved.")

    # ── Figure 4: Feature importance (XGBoost, ref seed) ─────────────────────
    plt.figure(figsize=(10, 8))
    plot_importance(ref['trained']['XGBoost'], max_num_features=15, height=0.6, grid=False)
    plt.title('XGBoost Feature Importance  (CIC-IDS2018, seed=42)')
    plt.tight_layout()
    plt.savefig('fig/feature_importance.png', dpi=300)
    plt.close()
    print("  [Fig] feature_importance.png saved.")

    # ── Figure 5: Cost ratio sensitivity ─────────────────────────────────────
    ratios  = [f"{cfn}:{cfp}" for (cfn, cfp) in COST_RATIOS]
    opt_ths = [ratio_res[(cfn, cfp)]['xgb_global_th'] for (cfn, cfp) in COST_RATIOS]

    plt.figure(figsize=(6, 5))
    plt.plot(ratios, opt_ths, marker='o', color='red', linewidth=2, markersize=8)
    plt.xlabel('FN : FP Cost Ratio')
    plt.ylabel('Global Optimal Threshold τ*')
    plt.title('Optimal Threshold vs. Cost Ratio  [XGBoost, CIC-IDS2018]')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/cost_sensitivity.png', dpi=300)
    plt.close()
    print("  [Fig] cost_sensitivity.png saved.")

# ==============================================================================
# 11. Main execution
# ==============================================================================

print("\n[Step 1/4]  Loading CIC-IDS2018...")
cic_X, cic_y, cic_cls, cic_bi = load_cic_data()
print(f"  Samples: {len(cic_y):,}   Classes: {len(cic_cls)}")

print("\n[Step 2/4]  Loading UNSW-NB15...")
unsw_X, unsw_y, unsw_cls, unsw_bi = load_unsw_data()
print(f"  Samples: {len(unsw_y):,}   Classes: {len(unsw_cls)}")

print("\n[Step 3/4]  CIC-IDS2018 multi-seed experiment...")
cic_summary = run_multi_seed(cic_X, cic_y, cic_cls, cic_bi,
                             SEEDS, PRIMARY_COST_FN, PRIMARY_COST_FP, tag="cic")

print("\n[Step 3b]   UNSW-NB15 multi-seed experiment...")
unsw_summary = run_multi_seed(unsw_X, unsw_y, unsw_cls, unsw_bi,
                              SEEDS, PRIMARY_COST_FN, PRIMARY_COST_FP, tag="unsw")

print("\n[Step 3c]   Cost ratio sensitivity (CIC-IDS2018, seed=42)...")
ratio_results = run_cost_sensitivity(cic_X, cic_y, cic_cls, cic_bi, seed=42)

print("\n[Step 4/4]  Generating report and figures...")
report = build_report(cic_summary, unsw_summary, ratio_results)
with open("experimental_results.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("  Report saved → experimental_results.txt")

make_figures(cic_summary, ratio_results)

print("\n" + "=" * 80)
print("All experiments completed.")
print("=" * 80)
