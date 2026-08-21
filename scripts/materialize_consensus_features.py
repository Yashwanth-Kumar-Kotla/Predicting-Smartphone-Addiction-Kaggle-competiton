"""
One-time materialization of the consensus-9 OpenFE feature set (exp070/071)
onto train/test, cached to disk. Every HPO/ensemble script since exp071 has
redundantly re-run OpenFE().fit() from scratch (~10-40 min depending on system
load) just to re-derive the same 9 features. Caching this once removes that
overhead from every future script.
"""
import time
import pickle
from openfe import OpenFE, transform

from common import load_data, NUM_COLS, TARGET, ID_COL


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def main():
    t0 = time.time()
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
    print(f"Matched {len(consensus_feats)}/{len(consensus_3_sigs)} consensus features")

    Xte = test[NUM_COLS].copy()
    X_new, Xte_new = transform(X, Xte, consensus_feats, n_jobs=8)
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    for c in new_cols:
        train[c] = X_new[c].values
        test[c] = Xte_new[c].values

    feat_cols = current_best_feats + new_cols
    print(f"Total feature count: {len(feat_cols)}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    with open("artifacts/consensus_data_cache.pkl", "wb") as f:
        pickle.dump({
            "train": train[[TARGET] + feat_cols],
            "test": test[[ID_COL] + feat_cols],
            "feat_cols": feat_cols,
        }, f)
    print("Saved artifacts/consensus_data_cache.pkl")


if __name__ == "__main__":
    main()
