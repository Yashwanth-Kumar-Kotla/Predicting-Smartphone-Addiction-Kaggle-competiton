"""
Experiment 015 — Phase 4, cheapest possible imputation test.

exp014's error analysis showed rows missing more of the 5 "real" features
have meaningfully higher error (0.061 -> 0.150 as missing count 0->5,
corr=0.075), motivating an imputation experiment. Start with the cheapest
version (per-fold median imputation, no leakage) before escalating to
model-based imputation -- if median imputation alone doesn't beat native
NaN handling, a fancier imputer is unlikely to be worth the extra compute.
"""
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

train, _ = load_data()
X, y = train[NUM_COLS], train[TARGET].values

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
    Xtr, Xva = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    ytr, yva = y[tr_idx], y[va_idx]

    medians = Xtr.median()  # fit imputer on TRAIN fold only, no leakage
    Xtr = Xtr.fillna(medians)
    Xva = Xva.fillna(medians)

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
print(f"\nvs exp008 native-NaN baseline (0.96461): delta = {oof_auc - 0.96461:+.5f}")

np.save("artifacts/oof_exp015_median_imputed.npy", oof)

log_experiment({
    "exp_id": "exp015",
    "model": "XGBoost",
    "features": "9 numeric only",
    "preprocessing": "per-fold median imputation (fit on train fold only)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": "cheapest imputation test, motivated by exp014 error analysis",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
