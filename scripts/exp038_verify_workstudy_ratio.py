"""
Experiment 038 — Verify a work_study_hours/daily_screen_time_hours ratio
before building anything on it, same discipline as exp031. Motivated by
the same discussion thread's finding that the original survey had
work_study > screen_time violations (9.1%) that the generator "fixed"
(enforced work_study <= screen_time), structurally analogous to the
entertainment_ratio constraint that gave a real, reproducible gain
(exp034-036).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, test = load_data()

sub = train[["daily_screen_time_hours", "work_study_hours", TARGET]].dropna().copy()
sub["ratio"] = sub["work_study_hours"] / sub["daily_screen_time_hours"]

violation = sub["work_study_hours"] > sub["daily_screen_time_hours"]
print(f"rows with valid ratio: {len(sub)}")
print(f"pct where work_study_hours > daily_screen_time_hours (violates the constraint): {violation.mean()*100:.2f}%")

print("\nbinned target rate by ratio (20 quantile bins):")
sub["bin"] = pd.qcut(sub["ratio"], q=20, duplicates="drop")
print(sub.groupby("bin", observed=True).agg(n=(TARGET, "size"), mean_ratio=("ratio", "mean"), target_rate=(TARGET, "mean")))

raw_auc = roc_auc_score(sub[TARGET], sub["ratio"])
print(f"\nraw-value AUC of ratio: {raw_auc:.5f}  (max(auc,1-auc)={max(raw_auc,1-raw_auc):.5f})")
print("(near 0.5 despite binned variation would indicate non-monotonicity, same pattern as entertainment_ratio)")

print("\ntest set: constraint check")
subt = test[["daily_screen_time_hours", "work_study_hours"]].dropna().copy()
violation_t = (subt["work_study_hours"] > subt["daily_screen_time_hours"])
print(f"test pct violating: {violation_t.mean()*100:.2f}%  (train was {violation.mean()*100:.2f}%)")
