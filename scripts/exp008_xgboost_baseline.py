"""
Experiment 008 — XGBoost baseline on the finalized 9-numeric-feature set
(categoricals dropped per exp006's clean CV ablation: -0.00002 AUC, noise).
Full 691k rows, 5-fold Stratified CV, native NaN handling.
"""
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

train, test = load_data()
X, y = train[NUM_COLS], train[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []

params = dict(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
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
print(f"best fold: {max(fold_aucs):.5f}  worst fold: {min(fold_aucs):.5f}")
print(f"runtime: {runtime:.1f}s")

np.save("artifacts/oof_exp008_xgboost_baseline.npy", oof)

log_experiment({
    "exp_id": "exp008",
    "model": "XGBoost",
    "features": "9 numeric only (categoricals dropped per exp006 ablation)",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": "third model family baseline, no tuning",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
