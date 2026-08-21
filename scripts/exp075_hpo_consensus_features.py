"""
Experiment 075 — Third HPO attempt, this time on the much smaller,
cleaner consensus-9 feature set (22 total features vs 53 before). Both
previous attempts (exp011 on 9 features, exp067 on 53 features) failed
because the search was undersized for the training cost -- with fewer
features here, training should be meaningfully faster, allowing more
trials in the same time budget. Real 5-fold CV per trial (no proxy).
"""
import time
import json
import pickle
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from openfe import OpenFE, transform

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train, test = load_data()
    train["social_ratio"] = train["social_media_hours"] / train["daily_screen_time_hours"]
    train["gaming_ratio"] = train["gaming_hours"] / train["daily_screen_time_hours"]
    train["entertainment_ratio"] = train["social_ratio"] + train["gaming_ratio"]
    train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
    current_best_feats = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]

    with open("artifacts/exp070_consensus_features.pkl", "rb") as f:
        consensus_data = pickle.load(f)
    consensus_3_sigs = set(consensus_data["consensus_3_signatures"])

    X = train[NUM_COLS].copy()
    y_df = train[[TARGET]].copy()
    ofe = OpenFE()
    features_seed42 = ofe.fit(data=X, label=y_df, n_jobs=8, seed=42, verbose=False)
    consensus_feats = [f for f in features_seed42 if signature(f) in consensus_3_sigs]

    Xte = test[NUM_COLS].copy()
    X_new, _ = transform(X, Xte, consensus_feats, n_jobs=8)
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    for c in new_cols:
        train[c] = X_new[c].values
    feat_cols = current_best_feats + new_cols
    print(f"Feature count for HPO: {len(feat_cols)}")

    Xd, y = train[feat_cols], train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(Xd, y))

    trial_oofs = {}

    def objective(trial):
        params = dict(
            n_estimators=6000,
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.09, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 3, 40, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
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

    N_TRIALS = 40
    TIMEOUT_SEC = 9000
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, show_progress_bar=False)
    runtime = time.time() - t0

    print(f"Completed {len(study.trials)} trials in {runtime:.1f}s")
    print(f"Best 5-fold CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    print(f"vs exp073 default-tuned XGBoost consensus (0.96721): delta = {study.best_value - 0.96721:+.5f}")

    print("\nTop 5 trials:")
    top5 = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
    for t in top5:
        print(f"  trial {t.number}: AUC={t.value:.5f}  params={t.params}")

    if study.best_trial.number in trial_oofs:
        best_oof = trial_oofs[study.best_trial.number]
        np.save("artifacts/oof_exp075_xgb_hpo_consensus.npy", best_oof)

    with open("artifacts/exp075_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved best OOF predictions and params.")


if __name__ == "__main__":
    main()
