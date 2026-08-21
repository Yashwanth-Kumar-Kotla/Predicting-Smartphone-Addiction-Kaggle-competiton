"""
Experiment 083 — exp081's CatBoost HPO only completed 6/20 trials in
3600s (CatBoost trials run ~3.5x slower than LightGBM's), which is below
Optuna TPESampler's default n_startup_trials=10 -- meaning the sampler
never left random exploration and never did real TPE-guided search. That
result is inconclusive, not a genuine "no headroom" finding. Retrying with
tighter bounds (lower iterations cap, higher depth ceiling removed, higher
learning_rate floor) for speed and a longer timeout to get a fair ~15-20
trial search, matching the standard that worked for XGBoost (exp075) and
LightGBM (exp079).
"""
import time
import json
import pickle
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from common import TARGET, N_FOLDS, SEED


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with open("artifacts/consensus_data_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"]
    feat_cols = cache["feat_cols"]
    print(f"Feature count for HPO: {len(feat_cols)}")

    Xd, y = train[feat_cols], train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(Xd, y))

    trial_oofs = {}

    def objective(trial):
        params = dict(
            iterations=2500,
            learning_rate=trial.suggest_float("learning_rate", 0.045, 0.13, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
            random_strength=trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            early_stopping_rounds=40,
            verbose=False,
            task_type="CPU",
        )
        oof = np.zeros(len(train))
        for tr_idx, va_idx in splits:
            Xtr, Xva = Xd.iloc[tr_idx], Xd.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            model = CatBoostClassifier(**params)
            model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
            oof[va_idx] = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y, oof)
        trial_oofs[trial.number] = oof
        return auc

    N_TRIALS = 20
    TIMEOUT_SEC = 7200
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, show_progress_bar=False)
    runtime = time.time() - t0

    print(f"Completed {len(study.trials)} trials in {runtime:.1f}s")
    print(f"Best 5-fold CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    print(f"vs CatBoost consensus-9 (exp072, untuned params, 0.96659): delta = {study.best_value - 0.96659:+.5f}")

    print("\nTop 5 trials:")
    top5 = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
    for t in top5:
        print(f"  trial {t.number}: AUC={t.value:.5f}  params={t.params}")

    if study.best_trial.number in trial_oofs:
        best_oof = trial_oofs[study.best_trial.number]
        np.save("artifacts/oof_exp083_catboost_hpo_consensus.npy", best_oof)

    with open("artifacts/exp083_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved best OOF predictions and params.")


if __name__ == "__main__":
    main()
