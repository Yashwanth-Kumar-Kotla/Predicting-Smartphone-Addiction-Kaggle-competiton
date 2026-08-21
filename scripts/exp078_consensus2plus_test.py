"""
Experiment 078 — exp071 showed the strict 3/3-seed consensus set (9
features) matches/beats the nested top-40 ceiling with zero selection
bias. Quick check: does the looser 2+/3-seed threshold (34 features)
do even better, or does it reintroduce some of the selection-bias noise
the strict threshold filtered out?
"""
import time
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
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
    consensus_2plus_sigs = set(consensus_data["consensus_2plus_signatures"])

    X = train[NUM_COLS].copy()
    y_df = train[[TARGET]].copy()
    ofe = OpenFE()
    features_seed42 = ofe.fit(data=X, label=y_df, n_jobs=8, seed=42, verbose=False)
    consensus_feats = [f for f in features_seed42 if signature(f) in consensus_2plus_sigs]
    print(f"Matched {len(consensus_feats)}/{len(consensus_2plus_sigs)} consensus-2+ features from seed=42's ranking")

    Xte = test[NUM_COLS].copy()
    X_new, _ = transform(X, Xte, consensus_feats, n_jobs=8)
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    for c in new_cols:
        train[c] = X_new[c].values
    feat_cols = current_best_feats + new_cols
    print(f"Total features: {len(feat_cols)}")

    y = train[TARGET].values
    with open("artifacts/exp075_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    t0 = time.time()
    for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
        Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof[va_idx] = model.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"\nConsensus-2+ (34 feat) XGBoost OOF AUC: {auc:.5f}  runtime: {time.time()-t0:.1f}s")
    print(f"vs consensus-9 (exp075 tuned, 0.96732): delta = {auc - 0.96732:+.5f}")


if __name__ == "__main__":
    main()
