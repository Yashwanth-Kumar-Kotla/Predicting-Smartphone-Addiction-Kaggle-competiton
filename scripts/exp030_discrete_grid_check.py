"""
Experiment 030 — Test the "discrete grid / template row" hypothesis raised
by a Kaggle discussion thread, WITHOUT the leaky target-encoding approach
they used (community-confirmed leakage: encoding computed outside the CV
loop). Hypothesis: the generator replicated exact value combinations from
a smaller seed dataset, so rows sharing identical values on the "weak"
features (sleep_hours, notifications_per_day, app_opens_per_day, age)
might cluster into groups with unusually consistent labels -- which would
explain exp006's finding that these features carry real joint signal
despite near-zero individual signal.

Diagnostic only, no leakage: checks cardinality and within-duplicate-group
target consistency using ALL of train (not CV-split), purely descriptive.
"""
import numpy as np
import pandas as pd
from common import load_data, NUM_COLS, TARGET

WEAK_NUM = ["sleep_hours", "notifications_per_day", "app_opens_per_day", "age"]
REAL5 = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
         "work_study_hours", "gaming_hours"]

train, test = load_data()

print("=" * 70)
print("CARDINALITY CHECK: are these features actually low-cardinality/discrete?")
print("=" * 70)
for c in NUM_COLS:
    n_unique = train[c].nunique()
    n_total = train[c].notna().sum()
    print(f"{c:28s}  unique={n_unique:7d}  non-null={n_total:7d}  ratio={n_unique/n_total:.4f}")

print("\n" + "=" * 70)
print("DUPLICATE-GROUP TARGET CONSISTENCY: weak features only")
print("=" * 70)
sub = train.dropna(subset=WEAK_NUM).copy()
group_sizes = sub.groupby(WEAK_NUM)[TARGET].transform("size")
sub["group_size"] = group_sizes
dup = sub[sub["group_size"] > 1]
print(f"rows with an exact-duplicate on all 4 weak features: {len(dup)} / {len(sub)} ({len(dup)/len(sub)*100:.2f}%)")

if len(dup) > 0:
    group_std = sub.groupby(WEAK_NUM)[TARGET].transform("std")
    dup_std = group_std[sub["group_size"] > 1]
    print(f"mean within-group target std (duplicate groups only): {dup_std.mean():.4f}")
    print(f"(for reference: std of a Bernoulli(0.709) variable = {np.sqrt(0.709*0.291):.4f} -- "
          f"if duplicate groups are much LOWER than this, that supports the template-row hypothesis)")

print("\n" + "=" * 70)
print("SAME CHECK ON THE 5 'REAL' FEATURES (sanity comparison)")
print("=" * 70)
for c in REAL5:
    n_unique = train[c].nunique()
    n_total = train[c].notna().sum()
    print(f"{c:28s}  unique={n_unique:7d}  non-null={n_total:7d}  ratio={n_unique/n_total:.4f}")

print("\n" + "=" * 70)
print("FULL 9-FEATURE ROW DUPLICATE CHECK (train vs itself, and train vs test)")
print("=" * 70)
full_dup_train = train[NUM_COLS].duplicated().sum()
print(f"exact duplicate rows on all 9 numeric features (train, ignoring NaN pattern differences): {full_dup_train}")

combined = pd.concat([train[NUM_COLS], test[NUM_COLS]], axis=0)
full_dup_combined = combined.duplicated().sum()
print(f"exact duplicate rows on all 9 numeric features (train+test combined): {full_dup_combined}")
