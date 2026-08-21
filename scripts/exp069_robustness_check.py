"""
Experiment 069 — Robustness check before adding more feature complexity.
v8's CV-to-LB gap shrank (+0.00077 vs the typical +0.0013-0.0018 seen all
session), a mild signal worth checking before pushing further. Two tests,
both cheap since they reuse exp050's already-computed full-data OpenFE
features (no re-discovery needed):

  (a) same seed=42 split, full-data-discovered top-40 features (vs
      exp061's per-fold-discovered features) -- isolates the full-data
      selection-bias effect specifically, same diagnostic as exp033 did
      for the ratio features.
  (b) a DIFFERENT fold seed (123) with the same full-data features --
      isolates whether the score depends heavily on which specific 20%
      happened to be held out (fold-specific overfitting check).
"""
import time
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from openfe import transform

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED


def main():
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

    X, y = train[feat_cols], train[TARGET].values

    with open("artifacts/exp022_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, n_jobs=-1)

    def run_cv(seed, name):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(train))
        fold_aucs = []
        t0 = time.time()
        for tr_idx, va_idx in skf.split(X, y):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            model = XGBClassifier(**{**params, "random_state": seed})
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = model.predict_proba(Xva)[:, 1]
            oof[va_idx] = pred
            fold_aucs.append(roc_auc_score(yva, pred))
        auc = roc_auc_score(y, oof)
        print(f"{name}: OOF AUC={auc:.5f}  fold_std={np.std(fold_aucs):.6f}  runtime={time.time()-t0:.1f}s")
        return auc

    auc_seed42_fulldata = run_cv(SEED, "seed=42 (matches exp061's fold split), full-data OpenFE features")
    auc_seed123_fulldata = run_cv(123, "seed=123 (different fold split), full-data OpenFE features")

    print(f"\nexp061 (nested, per-fold discovery, seed=42): 0.96720")
    print(f"this run (full-data discovery, seed=42): {auc_seed42_fulldata:.5f}  (isolates full-data selection-bias effect: {auc_seed42_fulldata - 0.96720:+.5f})")
    print(f"this run (full-data discovery, seed=123): {auc_seed123_fulldata:.5f}  (isolates fold-split stability: {auc_seed123_fulldata - auc_seed42_fulldata:+.5f})")


if __name__ == "__main__":
    main()
