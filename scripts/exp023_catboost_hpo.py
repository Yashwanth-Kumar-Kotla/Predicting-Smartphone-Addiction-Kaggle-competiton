"""
Experiment 023 — Properly-resourced CatBoost HPO (first real tuning pass).

CatBoost has never been tuned beyond fixing its iteration cap (exp019).
It's the weakest of the 3 tree baselines (0.96370 after convergence fix,
vs XGBoost's 0.96461-0.96473) using untuned depth=8 / l2_leaf_reg=3.0
defaults. Real 5-fold CV per trial (same fixed methodology as exp022 --
no 3-fold proxy), narrowed to CatBoost's typically-impactful knobs:
learning_rate, depth, l2_leaf_reg, bagging_temperature (Bayesian
bootstrap randomness), random_strength (split-scoring randomness).
"""
import time
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, _ = load_data()
X, y = train[NUM_COLS], train[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(X, y))

trial_oofs = {}


def objective(trial):
    params = dict(
        iterations=8000,
        learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 15.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        early_stopping_rounds=150,
        verbose=False,
        task_type="CPU",
    )

    oof = np.zeros(len(train))
    for tr_idx, va_idx in splits:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = CatBoostClassifier(**params)
        model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        oof[va_idx] = model.predict_proba(Xva)[:, 1]

    auc = roc_auc_score(y, oof)
    trial_oofs[trial.number] = oof
    return auc


N_TRIALS = 30
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
print(f"vs exp019 converged CatBoost baseline (0.96370): delta = {study.best_value - 0.96370:+.5f}")
print(f"vs exp022 tuned XGBoost (0.96473): delta = {study.best_value - 0.96473:+.5f}")

print("\nTop 5 trials:")
top5 = sorted(study.trials, key=lambda t: -t.value)[:5]
for t in top5:
    print(f"  trial {t.number}: AUC={t.value:.5f}  params={t.params}")

best_oof = trial_oofs[study.best_trial.number]
np.save("artifacts/oof_exp023_catboost_tuned.npy", best_oof)

import json
with open("artifacts/exp023_best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)

print("\nSaved best OOF predictions and params.")
