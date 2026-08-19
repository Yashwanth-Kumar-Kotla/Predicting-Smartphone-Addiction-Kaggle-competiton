"""
Experiment 045 — Does day_residual add anything ON TOP of our current
reference pipeline (9 raw + entertainment_ratio + workstudy_ratio,
exp042's ensemble = 0.96542)? exp044's raw AUC (0.847) is high but
suspect -- day_residual = 24 - daily_screen_time_hours - work_study_hours
- sleep_hours is dominated by daily_screen_time_hours (already 0.890 AUC
alone), and SUMS are comparatively easy for trees to approximate via
splits (unlike the ratios/division that gave real incremental gains).
Testing the actual incremental contribution on tuned XGBoost.
"""
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED

train, _ = load_data()
y = train[TARGET].values

with open("artifacts/exp022_best_params.json") as f:
    xgb_tuned = json.load(f)
params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
              early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

train["entertainment_ratio"] = (train["social_media_hours"] + train["gaming_hours"]) / train["daily_screen_time_hours"]
train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
train["day_residual"] = 24 - (train["daily_screen_time_hours"] + train["work_study_hours"] + train["sleep_hours"])
feat_cols = NUM_COLS + ["entertainment_ratio", "workstudy_ratio", "day_residual"]

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
print(f"OOF AUC (both ratios + day_residual): {auc:.5f}  fold_std: {np.std(fold_aucs):.6f}  runtime: {time.time()-t0:.1f}s")
print(f"vs exp040 both ratios only (0.96524): delta = {auc - 0.96524:+.5f}")

from common import log_experiment
log_experiment({
    "exp_id": "exp045",
    "model": "XGBoost (tuned) + both ratios + day_residual",
    "features": "9 numeric + entertainment_ratio + workstudy_ratio + day_residual",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}, same split as exp040",
    "cv_mean": f"{auc:.5f}", "cv_std": f"{np.std(fold_aucs):.6f}",
    "best_fold": f"{max(fold_aucs):.5f}", "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{time.time()-t0:.1f}",
    "notes": "incremental test of day_residual on top of the current reference pipeline (both ratios already included)",
    "conclusion": "TBD",
})
