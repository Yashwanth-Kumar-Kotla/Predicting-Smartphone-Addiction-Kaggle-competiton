"""
Experiment 012 — Confirm exp011's Optuna-found XGBoost hyperparameters with
a proper 5-fold CV (same protocol as exp008) for an apples-to-apples
comparison. exp011's 0.96410 was measured on 3-fold CV (less training data
per fold, not directly comparable to exp008's 5-fold 0.96461 baseline).
"""
import json
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

with open("artifacts/exp011_best_params.json") as f:
    best_params = json.load(f)

train, _ = load_data()
X, y = train[NUM_COLS], train[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []

params = dict(
    best_params,
    n_estimators=6000,
    tree_method="hist",
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=-1,
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
print(f"\nvs exp008 untuned baseline (0.96461): delta = {oof_auc - 0.96461:+.5f}")

np.save("artifacts/oof_exp012_xgboost_tuned.npy", oof)

log_experiment({
    "exp_id": "exp012",
    "model": "XGBoost (Optuna-tuned)",
    "features": "9 numeric only",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": "5-fold confirmation of exp011's Optuna best params (22 trials, 1hr budget)",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
