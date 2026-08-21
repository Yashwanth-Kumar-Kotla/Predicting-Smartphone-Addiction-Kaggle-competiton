"""
Experiment 072 — Apply the consensus-9 feature set (exp070/071's
breakthrough: matches nested-validated top-40 performance with zero
selection-bias gap) to LightGBM and CatBoost, completing the upgrade
across all 3 models. Uses standard (full-data) CV since the consensus
methodology already eliminates the selection-bias concern that made
nested CV necessary for the raw top-40/top-20 approach.
"""
import time
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from openfe import OpenFE, transform

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def main():
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
    print(f"Matched {len(consensus_feats)} consensus features")

    Xte = test[NUM_COLS].copy()
    X_new, _ = transform(X, Xte, consensus_feats, n_jobs=8)
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    for c in new_cols:
        train[c] = X_new[c].values
    feat_cols = current_best_feats + new_cols
    print(f"Total features: {len(feat_cols)}")

    y = train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print("\n" + "=" * 70 + "\nLightGBM + consensus-9 features\n" + "=" * 70)
    lgbm_params = dict(
        n_estimators=5000, learning_rate=0.03, num_leaves=63, max_depth=-1,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
        random_state=SEED, verbosity=-1,
    )
    oof_lgbm = np.zeros(len(train))
    t0 = time.time()
    for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
        Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = LGBMClassifier(**lgbm_params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                  callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        oof_lgbm[va_idx] = model.predict_proba(Xva)[:, 1]
    auc_lgbm = roc_auc_score(y, oof_lgbm)
    print(f"LightGBM OOF AUC: {auc_lgbm:.5f}  runtime: {time.time()-t0:.1f}s")
    print(f"vs LightGBM top-40 nested (exp063, 0.96655): delta = {auc_lgbm - 0.96655:+.5f}")
    np.save("artifacts/oof_exp072_lgbm_consensus.npy", oof_lgbm)

    print("\n" + "=" * 70 + "\nCatBoost (tuned) + consensus-9 features\n" + "=" * 70)
    with open("artifacts/exp023_best_params.json") as f:
        catboost_tuned = json.load(f)
    cb_params = dict(catboost_tuned, iterations=10000, loss_function="Logloss",
                      eval_metric="AUC", random_seed=SEED, early_stopping_rounds=150,
                      verbose=False, task_type="CPU")
    oof_cb = np.zeros(len(train))
    t0 = time.time()
    for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
        Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = CatBoostClassifier(**cb_params)
        model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        oof_cb[va_idx] = model.predict_proba(Xva)[:, 1]
    auc_cb = roc_auc_score(y, oof_cb)
    print(f"CatBoost OOF AUC: {auc_cb:.5f}  runtime: {time.time()-t0:.1f}s")
    print(f"vs CatBoost top-40 nested (exp064, 0.96674): delta = {auc_cb - 0.96674:+.5f}")
    np.save("artifacts/oof_exp072_catboost_consensus.npy", oof_cb)


if __name__ == "__main__":
    main()
