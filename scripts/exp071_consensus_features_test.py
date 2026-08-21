"""
Experiment 071 — Test the 9-feature consensus set (appeared in ALL 3
independent seeds' top-40, exp070) against the full top-40 set. If the
consensus set retains most of the performance with a much smaller
selection-bias gap (full-data CV closer to what a nested version would
show), that's a genuine win: fewer, more trustworthy features.
"""
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from openfe import OpenFE, transform
import json

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED


def main():
    train, test = load_data()
    train["social_ratio"] = train["social_media_hours"] / train["daily_screen_time_hours"]
    train["gaming_ratio"] = train["gaming_hours"] / train["daily_screen_time_hours"]
    train["entertainment_ratio"] = train["social_ratio"] + train["gaming_ratio"]
    train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
    current_best_feats = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]

    with open("artifacts/exp070_consensus_features.pkl", "rb") as f:
        consensus_data = pickle.load(f)

    # re-fit OpenFE on seed=42 (matches exp050's original discovery) to get
    # live Node objects for the consensus-3 signatures (pickled Node objects
    # from different seeds don't survive comparison cleanly across runs)
    X = train[NUM_COLS].copy()
    y_df = train[[TARGET]].copy()
    ofe = OpenFE()
    features_seed42 = ofe.fit(data=X, label=y_df, n_jobs=8, seed=42, verbose=False)

    def signature(feat):
        return (feat.name, tuple(sorted(feat.get_fnode())))

    consensus_3_sigs = set(consensus_data["consensus_3_signatures"])
    consensus_feats = [f for f in features_seed42 if signature(f) in consensus_3_sigs]
    print(f"Matched {len(consensus_feats)}/{len(consensus_3_sigs)} consensus-3 features from seed=42's ranking")

    Xte = test[NUM_COLS].copy()
    X_new, Xte_new = transform(X, Xte, consensus_feats, n_jobs=8)
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    for c in new_cols:
        train[c] = X_new[c].values
    feat_cols = current_best_feats + new_cols
    print(f"Total features (consensus set): {len(feat_cols)}")

    with open("artifacts/exp022_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, n_jobs=-1)

    def run_cv(seed, feats, name):
        Xd, yd = train[feats], train[TARGET].values
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(train))
        t0 = time.time()
        for tr_idx, va_idx in skf.split(Xd, yd):
            Xtr, Xva = Xd.iloc[tr_idx], Xd.iloc[va_idx]
            ytr, yva = yd[tr_idx], yd[va_idx]
            model = XGBClassifier(**{**params, "random_state": seed})
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            oof[va_idx] = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yd, oof)
        print(f"{name}: OOF AUC={auc:.5f}  runtime={time.time()-t0:.1f}s")
        return auc

    auc_consensus_seed42 = run_cv(SEED, feat_cols, "Consensus-9 features, seed=42")
    auc_consensus_seed123 = run_cv(123, feat_cols, "Consensus-9 features, seed=123")

    print(f"\nComparison:")
    print(f"  Full top-40, full-data-CV, seed=42 (exp069): 0.96669")
    print(f"  Full top-40, nested-CV, seed=42 (exp061):     0.96720")
    print(f"  Consensus-9, full-data-CV, seed=42:           {auc_consensus_seed42:.5f}")
    print(f"  Consensus-9, full-data-CV, seed=123:          {auc_consensus_seed123:.5f}")
    print(f"  Consensus-9 fold-stability (seed42 vs 123):   {auc_consensus_seed123 - auc_consensus_seed42:+.5f}")
    print(f"  Consensus-9 vs full-40 (both full-data, seed=42): {auc_consensus_seed42 - 0.96669:+.5f}")


if __name__ == "__main__":
    main()
