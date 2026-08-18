"""
Experiment 011 — Phase 6: Optuna hyperparameter optimization for XGBoost,
our current best single model (exp008, OOF AUC 0.96461, untuned defaults).

Uses a 3-fold CV (not 5) inside the objective to keep trial cost down --
Phase 13 compute-efficiency tradeoff: 3-fold OOF AUC is still a stable
enough proxy (given exp001's 5-fold std was ~0.0005, 3-fold should be in
a similar ballpark) to RANK trials against each other, even if the
absolute number differs slightly from 5-fold. The winning params get a
final 5-fold confirmation run afterward for the real logged number.

TPE sampler, median pruner to kill weak trials early using intermediate
per-fold validation AUC. Budget: 60 trials (~time-boxed, adjustable).
"""
import time
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, _ = load_data()
X, y = train[NUM_COLS], train[TARGET].values

TUNE_FOLDS = 3
skf_tune = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=SEED)
tune_splits = list(skf_tune.split(X, y))


def objective(trial):
    params = dict(
        n_estimators=2500,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=75,
        random_state=SEED,
        n_jobs=-1,
    )

    fold_aucs = []
    for fold_i, (tr_idx, va_idx) in enumerate(tune_splits):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, pred)
        fold_aucs.append(auc)
        trial.report(np.mean(fold_aucs), fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))


N_TRIALS = 60
TIMEOUT_SEC = 3600  # hard wall-clock cap regardless of trial count
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
)

t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, show_progress_bar=False)
runtime = time.time() - t0

print(f"Completed {len(study.trials)} trials in {runtime:.1f}s")
print(f"Best 3-fold CV AUC: {study.best_value:.5f}")
print(f"Best params: {study.best_params}")

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
print(f"completed trials: {len(completed)}  pruned trials: {len(pruned)}")

print("\nTop 5 trials:")
top5 = sorted(completed, key=lambda t: -t.value)[:5]
for t in top5:
    print(f"  trial {t.number}: AUC={t.value:.5f}  params={t.params}")

import json
with open("artifacts/exp011_best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)
print("\nSaved best params to artifacts/exp011_best_params.json")
