"""
Experiment 050 — Systematic automated feature discovery via OpenFE
(IIIS-Li-Group, arxiv 2211.12507), instead of continued manual guessing.
OpenFE searches ~23 transformation operators (arithmetic, ratios, GroupBy
aggregations, etc.) over feature combinations and ranks candidates by
their actual contribution to a GBDT model.

Goal: (1) see whether it independently rediscovers entertainment_ratio/
workstudy_ratio (would validate both OpenFE and our manual approach),
(2) surface genuinely new candidates we haven't tried.

Uses only the 9 numeric features (categoricals confirmed inert, exp006).

NOTE: OpenFE uses ProcessPoolExecutor internally, which on macOS requires
the spawn start method -- the whole script must be guarded by
`if __name__ == "__main__":` or child processes re-import and crash.
"""
import time
import pickle
import pandas as pd
from openfe import OpenFE, transform

from common import load_data, NUM_COLS, TARGET


def main():
    train, test = load_data()
    X = train[NUM_COLS].copy()
    y = train[[TARGET]].copy()
    Xte = test[NUM_COLS].copy()

    print(f"train shape: {X.shape}  test shape: {Xte.shape}")

    t0 = time.time()
    ofe = OpenFE()
    features = ofe.fit(data=X, label=y, n_jobs=8, seed=42, verbose=True)
    print(f"\nOpenFE discovery runtime: {time.time()-t0:.1f}s")
    print(f"Number of candidate features returned: {len(features)}")

    print("\nTop 30 discovered features (in ranked order):")
    for i, feat in enumerate(features[:30]):
        print(f"  {i+1}. {feat.name if hasattr(feat, 'name') else feat}")

    with open("artifacts/exp050_openfe_features.pkl", "wb") as f:
        pickle.dump(features, f)
    print("\nSaved discovered features to artifacts/exp050_openfe_features.pkl")


if __name__ == "__main__":
    main()
