"""
Experiment 079 — exp075 found a real, if small, HPO gain for XGBoost on
the consensus-9 feature set (+0.00011) now that training is fast enough
for a real search. LightGBM has never been tuned at all this session
(always used exp001's original defaults). Testing whether the same
tractable-HPO opportunity applies here.

Uses the cached consensus feature set (artifacts/consensus_data_cache.pkl,
built by materialize_consensus_features.py) instead of re-running OpenFE
discovery from scratch, which was taking 50+ min under current system load.
"""
import time
import json
import pickle
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

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
            n_estimators=2500,
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.09, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.6, 0.95),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 3.0, log=True),
            random_state=SEED,
            verbosity=-1,
            force_row_wise=True,
        )
        oof = np.zeros(len(train))
        for tr_idx, va_idx in splits:
            Xtr, Xva = Xd.iloc[tr_idx], Xd.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            model = LGBMClassifier(**params)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                      callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
            oof[va_idx] = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y, oof)
        trial_oofs[trial.number] = oof
        return auc

    N_TRIALS = 30
    TIMEOUT_SEC = 3600
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, show_progress_bar=False)
    runtime = time.time() - t0

    print(f"Completed {len(study.trials)} trials in {runtime:.1f}s")
    print(f"Best 5-fold CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    print(f"vs LightGBM consensus-9 default (exp072, 0.96653): delta = {study.best_value - 0.96653:+.5f}")

    print("\nTop 5 trials:")
    top5 = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
    for t in top5:
        print(f"  trial {t.number}: AUC={t.value:.5f}  params={t.params}")

    if study.best_trial.number in trial_oofs:
        best_oof = trial_oofs[study.best_trial.number]
        np.save("artifacts/oof_exp079_lgbm_hpo_consensus.npy", best_oof)

    with open("artifacts/exp079_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved best OOF predictions and params.")


if __name__ == "__main__":
    main()
