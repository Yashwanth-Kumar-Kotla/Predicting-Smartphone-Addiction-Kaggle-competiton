"""
Experiment 095 — exp075's XGBoost hyperparameters were tuned on the
22-feature consensus set. Since then the feature set has grown to 56
(consensus-9 + 14 artifacts + 20 expanded-consensus), and the community
research (discussion 734990) found model capacity and feature value
interact strongly -- undersized num_leaves/max_depth can make good
features look useless, and the reverse can happen too. Re-tuning XGBoost
on the current 56-feature set to check whether the old capacity settings
are still optimal, using the cached feature set (no rediscovery needed).
"""
import time
import json
import pickle
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import TARGET, N_FOLDS, SEED


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with open("artifacts/consensus_expanded_cache.pkl", "rb") as f:
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
            n_estimators=6000,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 40, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 3.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-6, 1.0, log=True),
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=100,
            random_state=SEED,
            n_jobs=-1,
        )
        oof = np.zeros(len(train))
        for tr_idx, va_idx in splits:
            Xtr, Xva = Xd.iloc[tr_idx], Xd.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            model = XGBClassifier(**params)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            oof[va_idx] = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y, oof)
        trial_oofs[trial.number] = oof
        return auc

    N_TRIALS = 25
    TIMEOUT_SEC = 9000
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, show_progress_bar=False)
    runtime = time.time() - t0

    print(f"Completed {len(study.trials)} trials in {runtime:.1f}s")
    print(f"Best 5-fold CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    print(f"vs exp092 old-params XGBoost on 56-feat set (0.96772): delta = {study.best_value - 0.96772:+.5f}")

    print("\nTop 5 trials:")
    top5 = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
    for t in top5:
        print(f"  trial {t.number}: AUC={t.value:.5f}  params={t.params}")

    if study.best_trial.number in trial_oofs:
        best_oof = trial_oofs[study.best_trial.number]
        np.save("artifacts/oof_exp095_xgb_hpo_expanded.npy", best_oof)

    with open("artifacts/exp095_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved best OOF predictions and params.")

    from common import log_experiment
    log_experiment({
        "exp_id": "exp095",
        "model": "XGBoost Optuna TPE re-tune on 56-feature expanded set (real 5-fold CV per trial)",
        "features": "consensus-9 + 14 artifacts + 20 expanded-consensus (56 total)",
        "preprocessing": "none (native NaN handling)",
        "hyperparams": f"search space: lr[0.02-0.09 log] max_depth[4-12] min_child_weight[1-40 log] subsample[0.6-0.95] colsample_bytree[0.3-1.0] reg_alpha[1e-6-2 log] reg_lambda[1e-6-3 log] gamma[1e-6-1 log]; n_estimators=6000 capped",
        "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV per trial, TPESampler seed=42",
        "cv_mean": f"{study.best_value:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{runtime:.1f}",
        "notes": "checking whether optimal capacity shifted after feature set grew from 22 to 56 features (per community finding on capacity/feature interactions)",
        "conclusion": "TBD",
    })
    print("\nLogged exp095.")


if __name__ == "__main__":
    main()
