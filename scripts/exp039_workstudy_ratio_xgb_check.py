"""
Experiment 039 — Quick single-model check of work_study_hours/
daily_screen_time_hours ratio on tuned XGBoost, before committing to the
full 3-model verification exp034-036 used. exp038's diagnostic showed this
ratio is mostly MONOTONIC (unlike entertainment_ratio's hump), so trees
likely already capture it via simple splits -- testing to confirm/refute
that prediction rather than assuming.
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

train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
feat_cols = NUM_COLS + ["workstudy_ratio"]

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
print(f"OOF AUC (9 feat + workstudy_ratio): {auc:.5f}  fold_std: {np.std(fold_aucs):.6f}  runtime: {time.time()-t0:.1f}s")
print(f"vs exp022 tuned XGBoost, no ratio (0.96473): delta = {auc - 0.96473:+.5f}")
print(f"vs exp034 tuned XGBoost + entertainment_ratio (0.96491): delta = {auc - 0.96491:+.5f}")

from common import log_experiment
log_experiment({
    "exp_id": "exp039",
    "model": "XGBoost (tuned) + workstudy_ratio",
    "features": "9 numeric + work_study_hours/daily_screen_time_hours",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}, same split as exp022/034",
    "cv_mean": f"{auc:.5f}", "cv_std": f"{np.std(fold_aucs):.6f}",
    "best_fold": f"{max(fold_aucs):.5f}", "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{time.time()-t0:.1f}",
    "notes": "quick single-model check before full 3-model verification, since exp038 diagnostic showed mostly-monotonic (unlike entertainment_ratio's hump)",
    "conclusion": "TBD",
})
