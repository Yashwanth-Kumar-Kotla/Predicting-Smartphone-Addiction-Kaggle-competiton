"""
Experiment 077 — Final submission using exp076's ensemble with tuned
XGBoost (exp075's HPO result) replacing default params. CV = 0.96743.
Multi-seed bagging (3 seeds) as established.
"""
import time
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from openfe import OpenFE, transform

from common import load_data, NUM_COLS, TARGET, ID_COL, SEED


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def main():
    train, test = load_data()
    for df in (train, test):
        df["social_ratio"] = df["social_media_hours"] / df["daily_screen_time_hours"]
        df["gaming_ratio"] = df["gaming_hours"] / df["daily_screen_time_hours"]
        df["entertainment_ratio"] = df["social_ratio"] + df["gaming_ratio"]
        df["workstudy_ratio"] = df["work_study_hours"] / df["daily_screen_time_hours"]
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
    X_new, Xte_new = transform(X, Xte, consensus_feats, n_jobs=8)
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    for c in new_cols:
        train[c] = X_new[c].values
        test[c] = Xte_new[c].values

    all_feat_cols = current_best_feats + new_cols
    print(f"Total feature count: {len(all_feat_cols)}")

    yfull = train[TARGET].values

    with open("artifacts/exp075_best_params.json") as f:
        xgb_tuned = json.load(f)
    with open("artifacts/exp023_best_params.json") as f:
        catboost_tuned = json.load(f)

    SEEDS = [42, 202, 2026]
    preds = {"lgbm": [], "catboost": [], "xgboost": []}

    t0 = time.time()
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        Xtr, Xval, ytr, yval = train_test_split(
            train[all_feat_cols], yfull, test_size=0.05, stratify=yfull, random_state=seed
        )

        print("LightGBM...")
        lgbm_params = dict(
            n_estimators=6000, learning_rate=0.03, num_leaves=63, max_depth=-1,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
            random_state=seed, verbosity=-1,
        )
        m = LGBMClassifier(**lgbm_params)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="auc",
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        preds["lgbm"].append(m.predict_proba(test[all_feat_cols])[:, 1])
        print(f"  best_iter={m.best_iteration_}")

        print("CatBoost...")
        cb_params = dict(catboost_tuned, iterations=10000, loss_function="Logloss",
                          eval_metric="AUC", random_seed=seed, early_stopping_rounds=150,
                          verbose=False, task_type="CPU")
        m = CatBoostClassifier(**cb_params)
        m.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
        preds["catboost"].append(m.predict_proba(test[all_feat_cols])[:, 1])
        print(f"  best_iter={m.get_best_iteration()}")

        print("XGBoost (HPO-tuned)...")
        xgb_params = dict(xgb_tuned, n_estimators=7000, tree_method="hist",
                           eval_metric="auc", early_stopping_rounds=100,
                           random_state=seed, n_jobs=-1)
        m = XGBClassifier(**xgb_params)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        preds["xgboost"].append(m.predict_proba(test[all_feat_cols])[:, 1])
        print(f"  best_iter={m.best_iteration}")

        print(f"seed {seed} done, elapsed {time.time()-t0:.0f}s")

    pred_lgbm = np.mean(preds["lgbm"], axis=0)
    pred_catboost = np.mean(preds["catboost"], axis=0)
    pred_xgb = np.mean(preds["xgboost"], axis=0)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

    np.save("artifacts/pred_test_exp077_lgbm.npy", pred_lgbm)
    np.save("artifacts/pred_test_exp077_catboost.npy", pred_catboost)
    np.save("artifacts/pred_test_exp077_xgboost.npy", pred_xgb)

    with open("artifacts/exp076_ensemble_weights.json") as f:
        weights = json.load(f)
    print(f"Blend weights (from exp076): {weights}")

    final_pred = (
        weights["lgbm"] * pred_lgbm
        + weights["catboost"] * pred_catboost
        + weights["xgboost"] * pred_xgb
    )
    final_pred = np.clip(final_pred, 0.0, 1.0)

    sample_sub = pd.read_csv("playground-series-s6e8/sample_submission.csv")
    assert len(final_pred) == len(test)
    assert (test[ID_COL].values == sample_sub[ID_COL].values).all()
    assert np.isfinite(final_pred).all()
    assert (final_pred >= 0).all() and (final_pred <= 1).all()

    submission = pd.DataFrame({"id": test[ID_COL].values, "addicted_label": final_pred})
    submission.to_csv("submission_v10.csv", index=False)

    old_sub = pd.read_csv("submission_v9.csv")
    diff = (submission["addicted_label"] - old_sub["addicted_label"]).abs()
    print(f"\nAll Phase 15 sanity checks passed.")
    print(f"submission_v10.csv written: {len(submission)} rows")
    print(f"mean |change| vs submission_v9.csv (prev): {diff.mean():.5f}")
    print(f"correlation with submission_v9.csv: {submission['addicted_label'].corr(old_sub['addicted_label']):.6f}")
    print(f"\nprediction stats: min={final_pred.min():.5f} max={final_pred.max():.5f} mean={final_pred.mean():.5f}")
    print(f"(train target mean for reference: {yfull.mean():.5f})")


if __name__ == "__main__":
    main()
