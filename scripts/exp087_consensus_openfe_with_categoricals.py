"""
Experiment 087 — exp086's single-seed nested holdout found a small
(+0.00010) positive signal from including categorical columns
(gender/stress_level/academic_work_impact) in OpenFE's candidate space,
with 2 GroupBy-style categorical interactions surfacing in the top-40.
That's too weak/noisy a signal from ONE split to trust on its own --
applying the same 3-seed consensus methodology that worked for the
numeric-only search (exp070/071, the session's biggest win) before
deciding whether this is real.
"""
import time
import pickle
from collections import Counter
from openfe import OpenFE

from common import load_data, NUM_COLS, CAT_COLS, TARGET


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def main():
    train, _ = load_data()
    disc_cols = NUM_COLS + CAT_COLS
    X = train[disc_cols].copy()
    y = train[[TARGET]].copy()

    SEEDS = [42, 123, 7]
    all_top40_signatures = []
    all_features_by_seed = {}

    for seed in SEEDS:
        t0 = time.time()
        ofe = OpenFE()
        features = ofe.fit(data=X, label=y, n_jobs=8, seed=seed, verbose=False,
                            categorical_features=CAT_COLS)
        top40 = features[:40]
        sigs = [signature(f) for f in top40]
        all_top40_signatures.append(set(sigs))
        all_features_by_seed[seed] = top40
        cat_involved = [s for s in sigs if any(c in s[1] for c in CAT_COLS)]
        print(f"seed {seed}: discovery took {time.time()-t0:.1f}s, {len(features)} total candidates, "
              f"{len(cat_involved)}/40 top involve a categorical")

    counter = Counter()
    for sigs in all_top40_signatures:
        counter.update(sigs)

    print("\nFeature consensus across 3 independent seeds' top-40 lists:")
    for count in [3, 2, 1]:
        feats_at_count = [sig for sig, c in counter.items() if c == count]
        n_cat = sum(1 for sig in feats_at_count if any(c in sig[1] for c in CAT_COLS))
        print(f"  appeared in exactly {count}/3 seeds' top-40: {len(feats_at_count)} features ({n_cat} involve a categorical)")

    consensus_3 = [sig for sig, c in counter.items() if c == 3]
    consensus_2plus = [sig for sig, c in counter.items() if c >= 2]
    print(f"\nFeatures in ALL 3 seeds (highest confidence): {len(consensus_3)}")
    for sig in consensus_3:
        tag = " [CATEGORICAL]" if any(c in sig[1] for c in CAT_COLS) else ""
        print(f"  {sig}{tag}")

    with open("artifacts/exp087_consensus_with_cat_features.pkl", "wb") as f:
        pickle.dump({
            "seed42_features": all_features_by_seed[42],
            "consensus_3_signatures": consensus_3,
            "consensus_2plus_signatures": consensus_2plus,
            "all_features_by_seed": {s: [signature(f) for f in feats] for s, feats in all_features_by_seed.items()},
        }, f)
    print("\nSaved to artifacts/exp087_consensus_with_cat_features.pkl")


if __name__ == "__main__":
    main()
