"""
Experiment 070 — exp069 showed full-data OpenFE selection carries a real
selection-bias cost (-0.00051 vs nested). Testing whether bagging the
feature-SELECTION step itself (multiple independent full-data OpenFE
fits with different seeds, keep only features selected consistently
across seeds) reduces this bias, the same way model-bagging reduces
training variance. Cheap: each full-data discovery only takes ~200-250s.
"""
import time
import pickle
from collections import Counter
from openfe import OpenFE

from common import load_data, NUM_COLS, TARGET


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def main():
    train, _ = load_data()
    X = train[NUM_COLS].copy()
    y = train[[TARGET]].copy()

    SEEDS = [42, 123, 7]
    all_top40_signatures = []
    all_features_by_seed = {}

    for seed in SEEDS:
        t0 = time.time()
        ofe = OpenFE()
        features = ofe.fit(data=X, label=y, n_jobs=8, seed=seed, verbose=False)
        top40 = features[:40]
        sigs = [signature(f) for f in top40]
        all_top40_signatures.append(set(sigs))
        all_features_by_seed[seed] = top40
        print(f"seed {seed}: discovery took {time.time()-t0:.1f}s, {len(features)} total candidates")

    counter = Counter()
    for sigs in all_top40_signatures:
        counter.update(sigs)

    print("\nFeature consensus across 3 independent seeds' top-40 lists:")
    for count in [3, 2, 1]:
        feats_at_count = [sig for sig, c in counter.items() if c == count]
        print(f"  appeared in exactly {count}/3 seeds' top-40: {len(feats_at_count)} features")

    consensus_3 = [sig for sig, c in counter.items() if c == 3]
    consensus_2plus = [sig for sig, c in counter.items() if c >= 2]
    print(f"\nFeatures in ALL 3 seeds (highest confidence): {len(consensus_3)}")
    for sig in consensus_3:
        print(f"  {sig}")
    print(f"\nFeatures in 2+ seeds: {len(consensus_2plus)}")

    with open("artifacts/exp070_consensus_features.pkl", "wb") as f:
        pickle.dump({
            "seed42_features": all_features_by_seed[42],
            "consensus_3_signatures": consensus_3,
            "consensus_2plus_signatures": consensus_2plus,
            "all_features_by_seed": {s: [signature(f) for f in feats] for s, feats in all_features_by_seed.items()},
        }, f)
    print("\nSaved consensus analysis to artifacts/exp070_consensus_features.pkl")


if __name__ == "__main__":
    main()
