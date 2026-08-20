"""
Experiment 053 — exp052 showed the 20-feature OpenFE gain (+0.00106) was
dominated by just 2 features: sqrt(daily_screen_time_hours) (importance
0.330, exceeding the raw feature) and daily_screen_time_hours +
weekend_screen_time (0.075). Both are explainable via histogram bin
resolution, not selection-bias noise. Everything else in the top 20 was
<0.02 importance. Testing the distilled 2-feature version with a proper
5-fold CV (not the single-split diagnostic) for a clean, trustworthy
number before committing to it.
"""
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

train, _ = load_data()
y = train[TARGET].values

train["social_ratio"] = train["social_media_hours"] / train["daily_screen_time_hours"]
train["gaming_ratio"] = train["gaming_hours"] / train["daily_screen_time_hours"]
train["entertainment_ratio"] = train["social_ratio"] + train["gaming_ratio"]
train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
train["sqrt_daily_screen_time"] = np.sqrt(train["daily_screen_time_hours"])
train["daily_plus_weekend"] = train["daily_screen_time_hours"] + train["weekend_screen_time"]

feat_cols = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio",
                          "sqrt_daily_screen_time", "daily_plus_weekend"]

with open("artifacts/exp022_best_params.json") as f:
    xgb_tuned = json.load(f)
params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
              early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []
t0 = time.time()
for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
    Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    model = XGBClassifier(**params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred = model.predict_proba(Xva)[:, 1]
    oof[va_idx] = pred
    fold_aucs.append(roc_auc_score(yva, pred))

auc = roc_auc_score(y, oof)
print(f"OOF AUC (current best + sqrt_daily + daily_plus_weekend): {auc:.5f}  fold_std: {np.std(fold_aucs):.6f}  runtime: {time.time()-t0:.1f}s")
print(f"vs exp046 Variant B current best (0.96538): delta = {auc - 0.96538:+.5f}")
print(f"vs exp051 full 20-feature batch (0.96644): delta = {auc - 0.96644:+.5f}")

np.save("artifacts/oof_exp053_distilled_xgb.npy", oof)

log_experiment({
    "exp_id": "exp053",
    "model": "XGBoost (tuned) + distilled OpenFE features (2 only)",
    "features": "current best (13) + sqrt_daily_screen_time + daily_plus_weekend",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{auc:.5f}", "cv_std": f"{np.std(fold_aucs):.6f}",
    "best_fold": f"{max(fold_aucs):.5f}", "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{time.time()-t0:.1f}",
    "notes": "clean 5-fold CV test of the 2 features that actually drove exp051's importance (sqrt + sum), dropping the other 18 low-importance OpenFE candidates",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
