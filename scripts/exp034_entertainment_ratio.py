"""
Experiment 034 — Test entertainment_ratio = (social_media_hours +
gaming_hours) / daily_screen_time_hours on the ACTUAL tuned XGBoost model
(exp022 params), not just a diagnostic. Unlike exp031's 2-variable
weekend/daily ratio (confirmed null for trees, both by the source thread
and our own exp010 principle), this is a 3-variable ratio -- genuinely
harder for a depth-5/6 tree to discover via sequential splits than a
2-variable one, so worth checking directly rather than assuming null.
"""
import time
import numpy as np
import json
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
feat_cols = NUM_COLS + ["entertainment_ratio"]

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
print(f"OOF AUC (9 feat + entertainment_ratio): {auc:.5f}  fold_std: {np.std(fold_aucs):.6f}  runtime: {time.time()-t0:.1f}s")
print(f"vs exp022 tuned XGBoost, no ratio (0.96473): delta = {auc - 0.96473:+.5f}")

np.save("artifacts/oof_exp034_entertainment_ratio.npy", oof)

from common import log_experiment
log_experiment({
    "exp_id": "exp034",
    "model": "XGBoost (tuned) + entertainment_ratio",
    "features": "9 numeric + (social_media_hours+gaming_hours)/daily_screen_time_hours",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}, same split as exp022",
    "cv_mean": f"{auc:.5f}", "cv_std": f"{np.std(fold_aucs):.6f}",
    "best_fold": f"{max(fold_aucs):.5f}", "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{time.time()-t0:.1f}",
    "notes": "3-variable ratio, harder for depth-5/6 trees to reconstruct than the 2-var ratio exp031/032 already ruled out",
    "conclusion": "TBD",
})
