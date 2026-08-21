"""
Experiment 091 — exp088/089 validated 14 generator-artifact features
(other_screen residual + decimal-fraction artifacts), giving a genuine
+0.00024 ensemble gain. Natural next question: does OpenFE find NEW
higher-order interactions combining the artifact features with the
existing numeric columns (e.g. other_screen x age, or GroupBy using
first_digit as a near-categorical key) that neither the original
consensus-9 search nor the raw artifact features alone captured?

Same 3-seed consensus methodology as exp070/087 (the only leak-safe way
to use automated discovery established this session).
"""
import time
import pickle
from collections import Counter
from openfe import OpenFE

from common import load_data, DATA_DIR, NUM_COLS, TARGET


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def add_arithmetic_artifact(df):
    df = df.copy()
    df["other_screen"] = df["daily_screen_time_hours"] - (
        df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]
    )
    df["other_screen_abs"] = df["other_screen"].abs()
    return df


def add_decimal_artifact(df, raw_str_df):
    import pandas as pd
    df = df.copy()
    decimal_cols = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
                     "work_study_hours", "sleep_hours", "weekend_screen_time"]
    for c in decimal_cols:
        s = raw_str_df[c].astype(str)
        dec_part = s.str.split(".").str[1]
        df[f"{c}_decimals_len"] = dec_part.str.len().fillna(0).astype(int)
        df[f"{c}_first_digit"] = dec_part.str[0].fillna("-1").astype(int)
    return df


def main():
    import pandas as pd
    train, _ = load_data()
    train_str = pd.read_csv(f"{DATA_DIR}/train.csv", dtype=str)
    train = add_decimal_artifact(add_arithmetic_artifact(train), train_str)
    artifact_cols = [c for c in train.columns if c not in NUM_COLS
                      and c not in ("id", "gender", "stress_level", "academic_work_impact", TARGET)]
    disc_cols = NUM_COLS + artifact_cols
    print(f"Discovery candidate columns ({len(disc_cols)}): {disc_cols}")

    X = train[disc_cols].copy()
    y = train[[TARGET]].copy()

    SEEDS = [42, 123, 7]
    all_top40_signatures = []
    all_features_by_seed = {}

    for seed in SEEDS:
        t0 = time.time()
        ofe = OpenFE()
        features = ofe.fit(data=X, label=y, n_jobs=4, seed=seed, verbose=False)
        top40 = features[:40]
        sigs = [signature(f) for f in top40]
        all_top40_signatures.append(set(sigs))
        all_features_by_seed[seed] = top40
        n_artifact_involved = sum(1 for s in sigs if any(c in s[1] for c in artifact_cols))
        print(f"seed {seed}: discovery took {time.time()-t0:.1f}s, {len(features)} total candidates, "
              f"{n_artifact_involved}/40 top involve an artifact column")

    counter = Counter()
    for sigs in all_top40_signatures:
        counter.update(sigs)

    print("\nFeature consensus across 3 independent seeds' top-40 lists:")
    for count in [3, 2, 1]:
        feats_at_count = [sig for sig, c in counter.items() if c == count]
        n_art = sum(1 for sig in feats_at_count if any(c in sig[1] for c in artifact_cols))
        print(f"  appeared in exactly {count}/3 seeds' top-40: {len(feats_at_count)} features ({n_art} involve an artifact col)")

    consensus_3 = [sig for sig, c in counter.items() if c == 3]
    print(f"\nFeatures in ALL 3 seeds (highest confidence): {len(consensus_3)}")
    for sig in consensus_3:
        tag = " [ARTIFACT]" if any(c in sig[1] for c in artifact_cols) else ""
        print(f"  {sig}{tag}")

    with open("artifacts/exp091_consensus_with_artifacts.pkl", "wb") as f:
        pickle.dump({
            "seed42_features": all_features_by_seed[42],
            "consensus_3_signatures": consensus_3,
            "all_features_by_seed": {s: [signature(f) for f in feats] for s, feats in all_features_by_seed.items()},
            "artifact_cols": artifact_cols,
        }, f)
    print("\nSaved to artifacts/exp091_consensus_with_artifacts.pkl")


if __name__ == "__main__":
    main()
