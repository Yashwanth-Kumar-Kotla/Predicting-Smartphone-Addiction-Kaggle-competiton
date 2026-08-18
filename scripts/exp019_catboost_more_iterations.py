"""
Experiment 019 — exp007's CatBoost baseline hit its 5000-iteration cap in
EVERY fold (best_iter 4979-4998/5000) without early-stopping, unlike
LightGBM (best_iter ~3000-3700/5000) and XGBoost (~4000-4500/5000) which
both converged with room to spare. This means CatBoost was likely
undertrained, not fairly compared. Raise the cap to 12000 with real early
stopping and see if it actually converges and beats exp007's 0.96347 --
possibly closing some of the gap to XGBoost's 0.96461.
"""
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

train, _ = load_data()
X, y = train[NUM_COLS], train[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []

params = dict(
    iterations=12000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    early_stopping_rounds=150,
    verbose=False,
    task_type="CPU",
)

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    model = CatBoostClassifier(**params)
    model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    pred = model.predict_proba(Xva)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(yva, pred)
    fold_aucs.append(auc)
    print(f"fold {fold}: AUC={auc:.5f}  best_iter={model.get_best_iteration()}  (cap=12000)")

runtime = time.time() - t0
oof_auc = roc_auc_score(y, oof)
fold_std = float(np.std(fold_aucs))

print(f"\nOOF AUC: {oof_auc:.5f}")
print(f"fold mean: {np.mean(fold_aucs):.5f}  fold std: {fold_std:.6f}")
print(f"runtime: {runtime:.1f}s")
print(f"\nvs exp007 undertrained CatBoost baseline (0.96347): delta = {oof_auc - 0.96347:+.5f}")
print(f"vs exp008 XGBoost (current best single, 0.96461): delta = {oof_auc - 0.96461:+.5f}")

np.save("artifacts/oof_exp019_catboost_more_iters.npy", oof)

log_experiment({
    "exp_id": "exp019",
    "model": "CatBoost (more iterations)",
    "features": "9 numeric only",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": "fix exp007's undertraining -- iterations 5000->12000, early_stopping 100->150",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
