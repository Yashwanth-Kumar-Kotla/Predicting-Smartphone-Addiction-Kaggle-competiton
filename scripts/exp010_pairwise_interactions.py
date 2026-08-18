"""
Experiment 010 — Phase 4 feature engineering, first targeted attempt.

exp009's SHAP interaction analysis showed the top 10 feature-pair
interactions in the winning XGBoost model (exp008, OOF AUC 0.96461) are
ALL among the same 5 "real" features identified in exp003
(daily_screen_time_hours, weekend_screen_time, social_media_hours,
work_study_hours, gaming_hours) -- consistent with exp004/005's finding
that explicit pairwise products recovered most of the tree-vs-linear gap.

Add all C(5,2)=10 pairwise products of these features (NaN-propagating,
same as raw features -- no imputation) to the 9-numeric-feature set and
re-run XGBoost with identical hyperparams to exp008 for a clean A/B.
"""
import time
from itertools import combinations
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

REAL5 = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
         "work_study_hours", "gaming_hours"]

train, test = load_data()

new_feats = []
for f1, f2 in combinations(REAL5, 2):
    col = f"{f1}_x_{f2}"
    train[col] = train[f1] * train[f2]
    new_feats.append(col)

all_feats = NUM_COLS + new_feats
X, y = train[all_feats], train[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []

params = dict(
    n_estimators=5000, learning_rate=0.03, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", eval_metric="auc", early_stopping_rounds=100,
    random_state=SEED, n_jobs=-1,
)

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    model = XGBClassifier(**params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred = model.predict_proba(Xva)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(yva, pred)
    fold_aucs.append(auc)
    print(f"fold {fold}: AUC={auc:.5f}  best_iter={model.best_iteration}")

runtime = time.time() - t0
oof_auc = roc_auc_score(y, oof)
fold_std = float(np.std(fold_aucs))

print(f"\nOOF AUC: {oof_auc:.5f}")
print(f"fold mean: {np.mean(fold_aucs):.5f}  fold std: {fold_std:.6f}")
print(f"runtime: {runtime:.1f}s")
print(f"\nvs exp008 baseline (0.96461): delta = {oof_auc - 0.96461:+.5f}")

np.save("artifacts/oof_exp010_xgboost_pairwise.npy", oof)

log_experiment({
    "exp_id": "exp010",
    "model": "XGBoost",
    "features": f"9 numeric + 10 pairwise products of the 5 real features ({len(all_feats)} total)",
    "preprocessing": "products computed on raw values, NaN-propagating, no imputation",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": "Phase 4 first targeted FE attempt, guided by exp009 SHAP interactions",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
