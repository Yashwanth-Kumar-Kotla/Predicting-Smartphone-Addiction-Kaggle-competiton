"""
Experiment 031 — Verify the weekend/daily screen-time ratio envelope claim
from a Kaggle discussion thread before building anything on it.

Claim: weekend_screen_time / daily_screen_time_hours has a "hump-shaped"
(non-monotone) relationship with the target -- high inside an envelope
[1.044, 1.965] (matching the original 7500-row survey), lower outside,
much lower beyond 2.5x. ~26% of competition rows fall outside the
envelope. Independently verify the target-rate pattern on our own data.
"""
import numpy as np
import pandas as pd
from common import load_data, TARGET

train, test = load_data()

sub = train[["daily_screen_time_hours", "weekend_screen_time", TARGET]].dropna().copy()
sub["ratio"] = sub["weekend_screen_time"] / sub["daily_screen_time_hours"]

LOW, HIGH = 1.044, 1.965
inside = (sub["ratio"] >= LOW) & (sub["ratio"] <= HIGH)
beyond_2_5x = sub["ratio"] > 2.5

print(f"rows with valid ratio: {len(sub)}")
print(f"pct inside envelope [{LOW},{HIGH}]: {inside.mean()*100:.1f}%")
print(f"pct outside envelope: {(~inside).mean()*100:.1f}%")
print(f"pct beyond 2.5x: {beyond_2_5x.mean()*100:.1f}%")

print(f"\ntarget rate inside envelope: {sub.loc[inside, TARGET].mean():.4f}")
print(f"target rate outside envelope: {sub.loc[~inside, TARGET].mean():.4f}")
print(f"target rate beyond 2.5x: {sub.loc[beyond_2_5x, TARGET].mean():.4f}")
print(f"overall base rate: {sub[TARGET].mean():.4f}")

print("\nbinned target rate by ratio (20 quantile bins) -- looking for a hump shape:")
sub["bin"] = pd.qcut(sub["ratio"], q=20, duplicates="drop")
print(sub.groupby("bin", observed=True).agg(n=(TARGET, "size"), mean_ratio=("ratio", "mean"), target_rate=(TARGET, "mean")))

print("\ntest set: same envelope check (no target available, just distribution)")
subt = test[["daily_screen_time_hours", "weekend_screen_time"]].dropna().copy()
subt["ratio"] = subt["weekend_screen_time"] / subt["daily_screen_time_hours"]
inside_t = (subt["ratio"] >= LOW) & (subt["ratio"] <= HIGH)
print(f"test pct inside envelope: {inside_t.mean()*100:.1f}%  (train was {inside.mean()*100:.1f}%)")

print("\nUnivariate AUC of the raw ratio alone (raw-value, direction-agnostic) -- sanity check for monotone-vs-nonmonotone:")
from sklearn.metrics import roc_auc_score
raw_auc = roc_auc_score(sub[TARGET], sub["ratio"])
print(f"raw-value AUC of ratio: {raw_auc:.5f}  (max(auc,1-auc)={max(raw_auc,1-raw_auc):.5f})")
print("(if this is near 0.5 despite target rate varying a lot across bins above, that CONFIRMS non-monotonicity -- a monotone score can't separate a hump)")
