"""
Experiment 014 — Phase 7 error analysis on the current best model (exp013
ensemble, OOF AUC 0.96476). Three failed levers in a row (exp010 FE,
exp011/012 HPO, exp013 ensembling all returned marginal/negative deltas)
means we should look at WHERE the model is wrong before guessing at the
next fix. Specifically checking the backlog's two leading hypotheses:
  (a) errors concentrated in rows with missing values (esp. the 5 "real"
      features) -> would support prioritizing imputation
  (b) errors concentrated in the ambiguous middle of the sigmoid curve
      (~6-9h daily_screen_time_hours) -> would support harder-to-fix
      irreducible label noise near the decision boundary, not missingness
"""
import numpy as np
import pandas as pd
from common import load_data, NUM_COLS, TARGET

REAL5 = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
         "work_study_hours", "gaming_hours"]

train, _ = load_data()
y = train[TARGET].values
oof = np.load("artifacts/oof_exp013_ensemble.npy")

# Error = squared error on probability (Brier-style), robust single-row metric
err = (oof - y) ** 2
train["_err"] = err
train["_oof"] = oof

print("=" * 70)
print("(a) Error vs missingness in the 5 REAL features specifically")
print("=" * 70)
n_missing_real5 = train[REAL5].isna().sum(axis=1)
tmp = pd.DataFrame({"n_missing_real5": n_missing_real5, "err": err})
print(tmp.groupby("n_missing_real5")["err"].agg(["size", "mean"]))

print("\n" + "=" * 70)
print("(a2) Error vs total missingness across all 9 numeric features")
print("=" * 70)
n_missing_all = train[NUM_COLS].isna().sum(axis=1)
tmp2 = pd.DataFrame({"n_missing_all": n_missing_all, "err": err})
print(tmp2.groupby("n_missing_all")["err"].agg(["size", "mean"]))

print("\n" + "=" * 70)
print("(b) Error vs daily_screen_time_hours value (is it worst near the sigmoid midpoint?)")
print("=" * 70)
sub = train[["daily_screen_time_hours", "_err"]].dropna()
sub["bin"] = pd.qcut(sub["daily_screen_time_hours"], q=15, duplicates="drop")
print(sub.groupby("bin", observed=True)["_err"].agg(["size", "mean"]))

print("\n" + "=" * 70)
print("(b2) Error vs the model's own predicted probability (calibration / boundary check)")
print("=" * 70)
sub2 = pd.DataFrame({"oof": oof, "err": err})
sub2["bin"] = pd.qcut(sub2["oof"], q=15, duplicates="drop")
print(sub2.groupby("bin", observed=True)["err"].agg(["size", "mean"]))

print("\n" + "=" * 70)
print("SUMMARY STATS")
print("=" * 70)
print(f"overall mean squared error (Brier): {err.mean():.5f}")
print(f"correlation(n_missing_real5, err): {np.corrcoef(n_missing_real5, err)[0,1]:.5f}")
print(f"correlation(n_missing_all, err): {np.corrcoef(n_missing_all, err)[0,1]:.5f}")
