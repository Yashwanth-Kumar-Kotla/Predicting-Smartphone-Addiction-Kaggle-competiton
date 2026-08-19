"""
Experiment 033 — Corrected version of exp032's logistic regression test.

exp032 compared against the wrong baseline: it replaced exp016's full
24-feature set (9 raw + 5 squared + 10 pairwise products) with a smaller
14-feature set (9 raw + 5 ratio-derived), so the -0.01119 delta mostly
reflected LOSING the pairwise interaction terms (esp.
daily_screen_time_hours x social_media_hours, the single strongest
interaction per exp009's SHAP analysis), not the ratio features hurting.

Correct test: add the 5 ratio-derived features (ratio, ratio_sq,
inside_envelope, dist_from_envelope, beyond_2_5x) ON TOP of exp016's full
29-feature set, isolating their marginal contribution cleanly.
"""
import time
from itertools import combinations
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED

REAL5 = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
         "work_study_hours", "gaming_hours"]
LOW, HIGH = 1.044, 1.965

train, _ = load_data()
y = train[TARGET].values


def engineer_full(df):
    out = df[NUM_COLS].copy()
    for c in REAL5:
        out[f"{c}_sq"] = df[c] ** 2
    for f1, f2 in combinations(REAL5, 2):
        out[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    ratio = df["weekend_screen_time"] / df["daily_screen_time_hours"]
    out["ratio"] = ratio
    out["ratio_sq"] = ratio ** 2
    out["inside_envelope"] = ((ratio >= LOW) & (ratio <= HIGH)).astype(float)
    out["dist_from_envelope"] = np.where(ratio < LOW, LOW - ratio, np.where(ratio > HIGH, ratio - HIGH, 0.0))
    out["beyond_2_5x"] = (ratio > 2.5).astype(float)
    return out


def engineer_no_ratio(df):
    out = df[NUM_COLS].copy()
    for c in REAL5:
        out[f"{c}_sq"] = df[c] ** 2
    for f1, f2 in combinations(REAL5, 2):
        out[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return out


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def run(engineer_fn, name):
    oof = np.zeros(len(train))
    fold_aucs = []
    for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
        Xtr_df = engineer_fn(train.iloc[tr_idx])
        Xva_df = engineer_fn(train.iloc[va_idx])
        ytr, yva = y[tr_idx], y[va_idx]
        medians = Xtr_df.median()
        Xtr_df = Xtr_df.fillna(medians)
        Xva_df = Xva_df.fillna(medians)
        scaler = StandardScaler().fit(Xtr_df)
        Xtr = scaler.transform(Xtr_df)
        Xva = scaler.transform(Xva_df)
        lr = LogisticRegression(max_iter=2000)
        lr.fit(Xtr, ytr)
        pred = lr.predict_proba(Xva)[:, 1]
        oof[va_idx] = pred
        fold_aucs.append(roc_auc_score(yva, pred))
    auc = roc_auc_score(y, oof)
    print(f"{name}: OOF AUC={auc:.5f}  fold_std={np.std(fold_aucs):.6f}  n_features={Xtr_df.shape[1]}")
    return oof, auc


print("Re-running exp016's exact feature set as a same-session baseline check:")
oof_base, auc_base = run(engineer_no_ratio, "24-feature (no ratio, matches exp016)")

print("\nAdding the 5 ratio-derived features on top:")
oof_ratio, auc_ratio = run(engineer_full, "29-feature (+ ratio-derived)")

print(f"\nDelta from adding ratio features: {auc_ratio - auc_base:+.5f}")

np.save("artifacts/oof_exp033_logistic_ratio_corrected.npy", oof_ratio)

tree_oof = np.load("artifacts/oof_exp024_ensemble_v2.npy")
corr = np.corrcoef(oof_ratio, tree_oof)[0, 1]
from scipy.optimize import minimize
def neg_auc(w):
    w = np.abs(w); w = w / w.sum()
    return -roc_auc_score(y, w[0] * tree_oof + w[1] * oof_ratio)
res = minimize(neg_auc, [0.8, 0.2], method="Nelder-Mead")
best_w = np.abs(res.x); best_w = best_w / best_w.sum()
blend_auc = roc_auc_score(y, best_w[0] * tree_oof + best_w[1] * oof_ratio)
print(f"\ncorr_with_trees={corr:.5f}  blend_weights=(tree={best_w[0]:.3f}, logistic={best_w[1]:.3f})  blend_auc={blend_auc:.5f}  gain_vs_0.96494={blend_auc-0.96494:+.5f}")

from common import log_experiment
log_experiment({
    "exp_id": "exp033",
    "model": "LogisticRegression, corrected ratio-feature ablation",
    "features": "24-feature exp016 set + 5 ratio-derived (29 total)",
    "preprocessing": "per-fold median imputation + StandardScaler, both fit on train fold only",
    "hyperparams": "default LogisticRegression, C=1.0",
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"no_ratio={auc_base:.5f} with_ratio={auc_ratio:.5f}",
    "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": "~5",
    "notes": f"corrects exp032's confounded comparison; corr_with_trees={corr:.5f}; blend_auc={blend_auc:.5f}",
    "conclusion": "TBD",
})
