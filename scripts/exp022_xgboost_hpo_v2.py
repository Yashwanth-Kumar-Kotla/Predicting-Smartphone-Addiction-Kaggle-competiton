"""
Experiment 022 — Properly-resourced XGBoost HPO, take 2.

exp011/012 failed because it tuned on a 3-fold proxy that didn't
rank-correlate with the real 5-fold protocol (best 3-fold trial scored
WORSE on 5-fold confirmation than the untuned baseline). Fixing that here:
evaluate every trial with the REAL 5-fold StratifiedKFold protocol
directly -- more expensive per trial, but the reported value IS the
honest number, no separate confirmation run needed.

Also narrowing the search space around the region exp011's top trials
already converged to (max_depth 5-8, lr 0.02-0.075, min_child_weight
9-27, colsample_bytree 0.57-0.7, near-zero regularization) rather than
re-exploring the full space blindly -- more efficient use of a bigger but
still finite time budget.

Per-trial OOF arrays are cached in memory; only the best trial's OOF is
persisted to disk at the end (avoids a redundant refit).
"""
import time
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, _ = load_data()
X, y = train[NUM_COLS], train[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(X, y))

trial_oofs = {}


def objective(trial):
    params = dict(
        n_estimators=6000,
        learning_rate=trial.suggest_float("learning_rate", 0.015, 0.09, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 9),
        min_child_weight=trial.suggest_float("min_child_weight", 5, 35, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 0.9),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.8),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 2.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-6, 1.0, log=True),
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=SEED,
        n_jobs=-1,
    )

    oof = np.zeros(len(train))
    for tr_idx, va_idx in splits:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof[va_idx] = model.predict_proba(Xva)[:, 1]

    auc = roc_auc_score(y, oof)
    trial_oofs[trial.number] = oof
    return auc


N_TRIALS = 40
TIMEOUT_SEC = 9000  # 2.5 hours hard cap
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)

t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, show_progress_bar=False)
runtime = time.time() - t0

print(f"Completed {len(study.trials)} trials in {runtime:.1f}s")
print(f"Best 5-fold CV AUC (honest, no proxy): {study.best_value:.5f}")
print(f"Best params: {study.best_params}")
print(f"vs exp008 untuned baseline (0.96461): delta = {study.best_value - 0.96461:+.5f}")

print("\nTop 5 trials:")
top5 = sorted(study.trials, key=lambda t: -t.value)[:5]
for t in top5:
    print(f"  trial {t.number}: AUC={t.value:.5f}  params={t.params}")

best_oof = trial_oofs[study.best_trial.number]
np.save("artifacts/oof_exp022_xgboost_tuned_v2.npy", best_oof)

import json
with open("artifacts/exp022_best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)

print("\nSaved best OOF predictions and params.")
