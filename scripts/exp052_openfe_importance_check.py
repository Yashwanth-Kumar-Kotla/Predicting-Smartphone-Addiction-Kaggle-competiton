"""
Experiment 052 — exp051's +0.00106 gain from 20 OpenFE-discovered
features is unusually large (2x anything else found this session) and
was selected using the full training label column, raising real selection-
bias risk. Before trusting it: (1) check feature importance to see which
specific autoFE columns are actually driving the gain -- are they
sensible or arbitrary? (2) quick single-fit check is cheap; full
cross-model verification follows if this looks legitimate.
"""
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from openfe import transform

from common import load_data, NUM_COLS, TARGET, SEED


def main():
    train, test = load_data()
    X = train[NUM_COLS].copy()
    Xte = test[NUM_COLS].copy()

    with open("artifacts/exp050_openfe_features.pkl", "rb") as f:
        features = pickle.load(f)

    print("Top 20 discovered features (operator + columns):")
    for i, feat in enumerate(features[:20]):
        print(f"  autoFE_f_{i}: {feat.name}({feat.get_fnode()})")

    X_new, _ = transform(X, Xte, features[:20], n_jobs=8)
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    for c in new_cols:
        train[c] = X_new[c].values

    train["social_ratio"] = train["social_media_hours"] / train["daily_screen_time_hours"]
    train["gaming_ratio"] = train["gaming_hours"] / train["daily_screen_time_hours"]
    train["entertainment_ratio"] = train["social_ratio"] + train["gaming_ratio"]
    train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]

    feat_cols = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"] + new_cols
    y = train[TARGET].values

    Xtr, Xval, ytr, yval = train_test_split(train[feat_cols], y, test_size=0.1, stratify=y, random_state=SEED)

    with open("artifacts/exp022_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
    model = XGBClassifier(**params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

    importances = model.feature_importances_
    imp_df = list(zip(feat_cols, importances))
    imp_df.sort(key=lambda x: -x[1])
    print("\nFeature importances (single 90/10 split, for diagnostic purposes only):")
    for name, imp in imp_df:
        marker = " <-- autoFE" if name.startswith("autoFE") else ""
        print(f"  {name:30s} {imp:.5f}{marker}")


if __name__ == "__main__":
    main()
