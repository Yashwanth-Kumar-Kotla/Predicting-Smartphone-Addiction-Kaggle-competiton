"""
Experiment 067 — Our XGBoost hyperparameters (exp022) were tuned on the
original 9-feature set. The feature landscape has since grown to 53
features (ratios + top-40 OpenFE). Re-running HPO on this new landscape,
since optimal depth/regularization/colsample likely shifted with 5x more
features available at each split.

Uses the FULL-DATA OpenFE discovery (exp050, already validated via nested
CV as a procedure) as a FIXED feature set for the HPO search -- standard
CV during hyperparameter search on a fixed feature set doesn't introduce
new leakage (the feature-selection leakage risk was already handled
separately when we validated the procedure via nested CV).
"""
import time
import pickle
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from openfe import transform

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train, test = load_data()
    train["social_ratio"] = train["social_media_hours"] / train["daily_screen_time_hours"]
    train["gaming_ratio"] = train["gaming_hours"] / train["daily_screen_time_hours"]
    train["entertainment_ratio"] = train["social_ratio"] + train["gaming_ratio"]
    train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
    current_best_feats = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]

    with open("artifacts/exp050_openfe_features.pkl", "rb") as f:
        openfe_features = pickle.load(f)
    X_openfe, _ = transform(train[NUM_COLS].copy(), test[NUM_COLS].copy(), openfe_features[:40], n_jobs=8)
    openfe_cols = [c for c in X_openfe.columns if c not in NUM_COLS]
    for c in openfe_cols:
        train[c] = X_openfe[c].values

    feat_cols = current_best_feats + openfe_cols
    print(f"Feature count for HPO: {len(feat_cols)}")

    X, y = train[feat_cols], train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y))

    trial_oofs = {}

    def objective(trial):
        params = dict(
            n_estimators=6000,
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.09, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 3, 40, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 0.9),
            colsample_bynode=trial.suggest_float("colsample_bynode", 0.3, 1.0),
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
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            model = XGBClassifier(**params)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            oof[va_idx] = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y, oof)
        trial_oofs[trial.number] = oof
        return auc

    N_TRIALS = 30
    TIMEOUT_SEC = 9000
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, show_progress_bar=False)
    runtime = time.time() - t0

    print(f"Completed {len(study.trials)} trials in {runtime:.1f}s")
    print(f"Best 5-fold CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    print(f"vs exp061 default-tuned XGBoost on this feature set (0.96720): delta = {study.best_value - 0.96720:+.5f}")

    print("\nTop 5 trials:")
    top5 = sorted(study.trials, key=lambda t: -t.value)[:5]
    for t in top5:
        print(f"  trial {t.number}: AUC={t.value:.5f}  params={t.params}")

    best_oof = trial_oofs[study.best_trial.number]
    np.save("artifacts/oof_exp067_xgb_hpo_new_features.npy", best_oof)

    import json
    with open("artifacts/exp067_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved best OOF predictions and params.")


if __name__ == "__main__":
    main()
