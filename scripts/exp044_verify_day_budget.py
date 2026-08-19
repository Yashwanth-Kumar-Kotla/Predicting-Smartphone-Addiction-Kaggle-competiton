"""
Experiment 044 — Verify a "day budget residual" feature before building
anything on it: 24 - (daily_screen_time_hours + work_study_hours +
sleep_hours) = unaccounted hours in the day. Same reasoning family as
entertainment_ratio/workstudy_ratio (both winners), but genuinely
different -- uses sleep_hours (not in any ratio yet) and 3 raw features
in an additive/residual form rather than a ratio, so it's not collinear
with the two ratios already in the pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, test = load_data()

sub = train[["daily_screen_time_hours", "work_study_hours", "sleep_hours", TARGET]].dropna().copy()
sub["day_residual"] = 24 - (sub["daily_screen_time_hours"] + sub["work_study_hours"] + sub["sleep_hours"])

print(f"rows with all 3 components present: {len(sub)}")
print(f"day_residual stats: min={sub['day_residual'].min():.2f} max={sub['day_residual'].max():.2f} mean={sub['day_residual'].mean():.2f}")
print(f"pct negative (accounted hours exceed 24, an impossible day): {(sub['day_residual']<0).mean()*100:.2f}%")

print("\nbinned target rate by day_residual (20 quantile bins):")
sub["bin"] = pd.qcut(sub["day_residual"], q=20, duplicates="drop")
print(sub.groupby("bin", observed=True).agg(n=(TARGET, "size"), mean_resid=("day_residual", "mean"), target_rate=(TARGET, "mean")))

raw_auc = roc_auc_score(sub[TARGET], sub["day_residual"])
print(f"\nraw-value AUC of day_residual: {raw_auc:.5f}  (max(auc,1-auc)={max(raw_auc,1-raw_auc):.5f})")

print("\ntest set: same check")
subt = test[["daily_screen_time_hours", "work_study_hours", "sleep_hours"]].dropna().copy()
subt["day_residual"] = 24 - (subt["daily_screen_time_hours"] + subt["work_study_hours"] + subt["sleep_hours"])
print(f"test pct negative: {(subt['day_residual']<0).mean()*100:.2f}%  (train was {(sub['day_residual']<0).mean()*100:.2f}%)")
print(f"test day_residual mean: {subt['day_residual'].mean():.2f}  (train was {sub['day_residual'].mean():.2f})")
