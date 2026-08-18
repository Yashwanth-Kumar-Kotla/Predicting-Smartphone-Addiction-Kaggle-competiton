"""
Experiment 001 — First meaningful baseline: LightGBM, native categorical +
native missing-value handling, no feature engineering, 5-fold Stratified CV.
"""
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from common import load_data, FEATURE_COLS, CAT_COLS, TARGET, N_FOLDS, SEED, log_experiment

train, test = load_data()
X, y = train[FEATURE_COLS], train[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []
importances = np.zeros(len(FEATURE_COLS))

params = dict(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=SEED,
    verbosity=-1,
)

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    model = LGBMClassifier(**params)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    pred = model.predict_proba(Xva)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(yva, pred)
    fold_aucs.append(auc)
    importances += model.feature_importances_ / N_FOLDS
    print(f"fold {fold}: AUC={auc:.5f}  best_iter={model.best_iteration_}")

runtime = time.time() - t0
oof_auc = roc_auc_score(y, oof)
fold_std = float(np.std(fold_aucs))

print(f"\nOOF AUC: {oof_auc:.5f}")
print(f"fold mean: {np.mean(fold_aucs):.5f}  fold std: {fold_std:.6f}")
print(f"best fold: {max(fold_aucs):.5f}  worst fold: {min(fold_aucs):.5f}")
print(f"runtime: {runtime:.1f}s")

print("\nFeature importances (gain-normalized split count avg):")
for f, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    print(f"  {f:30s} {imp:.1f}")

np.save("artifacts/oof_exp001_lgbm_baseline.npy", oof)

log_experiment({
    "exp_id": "exp001",
    "model": "LightGBM",
    "features": "all raw (9 numeric + 3 categorical, native handling)",
    "preprocessing": "none (native NaN + categorical handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": "first baseline, no FE, no tuning",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
