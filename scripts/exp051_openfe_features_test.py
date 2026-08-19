"""
Experiment 051 — Use OpenFE's transform() to materialize its top-ranked
discovered features (exp050) and test them as a batch on top of our
current best feature set (9 raw + 4 ratio-derived), via tuned XGBoost CV.
Batch test first (efficient use of the tool's ranking); if positive,
decompose to find which specific features drive it.
"""
import time
import pickle
import numpy as np
import pandas as pd
from openfe import transform
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import json

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment


def main():
    train, test = load_data()
    X = train[NUM_COLS].copy()
    Xte = test[NUM_COLS].copy()

    with open("artifacts/exp050_openfe_features.pkl", "rb") as f:
        features = pickle.load(f)

    TOP_N = 20
    t0 = time.time()
    X_new, Xte_new = transform(X, Xte, features[:TOP_N], n_jobs=8)
    print(f"transform runtime: {time.time()-t0:.1f}s")
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    print(f"new columns generated: {new_cols}")

    for c in new_cols:
        train[c] = X_new[c].values

    for df in (train,):
        df["social_ratio"] = df["social_media_hours"] / df["daily_screen_time_hours"]
        df["gaming_ratio"] = df["gaming_hours"] / df["daily_screen_time_hours"]
        df["entertainment_ratio"] = df["social_ratio"] + df["gaming_ratio"]
        df["workstudy_ratio"] = df["work_study_hours"] / df["daily_screen_time_hours"]

    current_best_feats = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]
    feat_cols = current_best_feats + new_cols
    print(f"total features: {len(feat_cols)}")

    y = train[TARGET].values
    with open("artifacts/exp022_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    fold_aucs = []
    t0 = time.time()
    for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
        Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred = model.predict_proba(Xva)[:, 1]
        oof[va_idx] = pred
        fold_aucs.append(roc_auc_score(yva, pred))

    auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC (current best + top {TOP_N} OpenFE features): {auc:.5f}  fold_std: {np.std(fold_aucs):.6f}  runtime: {time.time()-t0:.1f}s")
    print(f"vs exp046 Variant B current best (0.96538): delta = {auc - 0.96538:+.5f}")

    np.save("artifacts/oof_exp051_openfe_batch.npy", oof)

    log_experiment({
        "exp_id": "exp051",
        "model": "XGBoost (tuned) + current best features + top 20 OpenFE features",
        "features": f"current best (13) + {len(new_cols)} OpenFE-discovered: {new_cols}",
        "preprocessing": "none (native NaN handling)",
        "hyperparams": str(params),
        "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
        "cv_mean": f"{auc:.5f}", "cv_std": f"{np.std(fold_aucs):.6f}",
        "best_fold": f"{max(fold_aucs):.5f}", "worst_fold": f"{min(fold_aucs):.5f}",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": "batch test of OpenFE-discovered features on top of current best pipeline",
        "conclusion": "TBD",
    })
    print("\nLogged to experiments/experiment_log.csv")


if __name__ == "__main__":
    main()
