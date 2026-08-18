"""
Experiment 002 — Cheap diagnostic (no model training): bin the top
univariate features and look at target rate per bin. If the generator used
a threshold or simple monotonic formula, this will show it directly as a
step function or clean monotonic curve rather than smooth noise.
"""
import numpy as np
import pandas as pd
from common import load_data, TARGET

pd.set_option("display.max_rows", 30)
train, _ = load_data()

TOP_FEATURES = [
    "daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
    "notifications_per_day", "app_opens_per_day",
]

for feat in TOP_FEATURES:
    sub = train[[feat, TARGET]].dropna()
    sub["bin"] = pd.qcut(sub[feat], q=20, duplicates="drop")
    agg = sub.groupby("bin", observed=True).agg(
        n=(TARGET, "size"),
        mean_val=(feat, "mean"),
        target_rate=(TARGET, "mean"),
    )
    print(f"\n{'=' * 70}\n{feat} — target rate by 20-quantile bin\n{'=' * 70}")
    print(agg.to_string())

print("\n" + "=" * 70)
print("ROW-LEVEL MISSING COUNT vs TARGET")
print("=" * 70)
feature_cols_all = [c for c in train.columns if c not in ("id", TARGET)]
n_missing = train[feature_cols_all].isna().sum(axis=1)
tmp = pd.DataFrame({"n_missing": n_missing, TARGET: train[TARGET]})
print(tmp.groupby("n_missing")[TARGET].agg(["size", "mean"]))
print(f"\ncorrelation(n_missing, target) = {tmp['n_missing'].corr(tmp[TARGET]):.5f}")

print("\n" + "=" * 70)
print("2D interaction probe: daily_screen_time_hours x social_media_hours")
print("=" * 70)
sub = train[["daily_screen_time_hours", "social_media_hours", TARGET]].dropna()
sub["bin_a"] = pd.qcut(sub["daily_screen_time_hours"], q=5, duplicates="drop")
sub["bin_b"] = pd.qcut(sub["social_media_hours"], q=5, duplicates="drop")
pivot = sub.pivot_table(index="bin_a", columns="bin_b", values=TARGET, aggfunc="mean")
print(pivot.to_string())
