"""
Experiment 005 — Isolate why exp004's 5-feature logistic regression
(AUC 0.92735) trails the full 12-feature LightGBM baseline (AUC 0.96384,
exp001) by 3.6 points. Two confounds to separate:

  (a) population: logistic reg only saw the 395,959 complete-case rows
      (57% of data) -- is that subset just intrinsically easier/harder?
  (b) missing structure: does LGBM on ONLY the 5 "real" features (native
      NaN handling, full 691k rows) already close most of the gap, or do
      the 7 "noise" features actually carry real joint/interaction signal?
  (c) interactions: do explicit pairwise products (not just squares) close
      more of the gap for the logistic model specifically?
"""
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from itertools import combinations
from common import load_data, FEATURE_COLS, CAT_COLS, TARGET, N_FOLDS, SEED

REAL_FEATURES = [
    "daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
    "work_study_hours", "gaming_hours",
]

train, _ = load_data()
y_full = train[TARGET].values
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_params = dict(
    n_estimators=5000, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, verbosity=-1,
)

# (b) LGBM using ONLY the 5 real features, full data, native NaN handling
print("=" * 70)
print("(b) LGBM on 5 real features only, full 691k rows, native NaN handling")
print("=" * 70)
X5 = train[REAL_FEATURES]
oof_5feat = np.zeros(len(train))
t0 = time.time()
for tr_idx, va_idx in skf.split(X5, y_full):
    model = LGBMClassifier(**lgb_params)
    model.fit(
        X5.iloc[tr_idx], y_full[tr_idx],
        eval_set=[(X5.iloc[va_idx], y_full[va_idx])], eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    oof_5feat[va_idx] = model.predict_proba(X5.iloc[va_idx])[:, 1]
auc_5feat_full = roc_auc_score(y_full, oof_5feat)
print(f"LGBM(5 real features, all rows) OOF AUC = {auc_5feat_full:.5f}  ({time.time()-t0:.0f}s)")
print(f"vs exp001 full 12-feature LGBM = 0.96384  (delta = {auc_5feat_full - 0.96384:+.5f})")

# (a) Restrict to the SAME complete-case subset used in exp004, compare
# LGBM (12 feat) vs LGBM (5 feat) vs logistic (5 feat, linear) on identical rows
print("\n" + "=" * 70)
print("(a)+(c) Apples-to-apples on the exp004 complete-case subset (5-feature complete cases)")
print("=" * 70)
mask = train[REAL_FEATURES].notna().all(axis=1)
sub = train.loc[mask].reset_index(drop=True)
y = sub[TARGET].values
print(f"subset size: {len(sub)} rows ({len(sub)/len(train)*100:.1f}% of train)")

skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# LGBM, all 12 features, but only on this subset
X12_sub = sub[FEATURE_COLS]
oof_12_sub = np.zeros(len(sub))
for tr_idx, va_idx in skf2.split(X12_sub, y):
    model = LGBMClassifier(**lgb_params)
    model.fit(
        X12_sub.iloc[tr_idx], y[tr_idx],
        eval_set=[(X12_sub.iloc[va_idx], y[va_idx])], eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    oof_12_sub[va_idx] = model.predict_proba(X12_sub.iloc[va_idx])[:, 1]
auc_12_sub = roc_auc_score(y, oof_12_sub)
print(f"LGBM(12 features) on complete-case subset OOF AUC = {auc_12_sub:.5f}")

# LGBM, only the 5 real features, on this subset
X5_sub = sub[REAL_FEATURES]
oof_5_sub = np.zeros(len(sub))
for tr_idx, va_idx in skf2.split(X5_sub, y):
    model = LGBMClassifier(**lgb_params)
    model.fit(
        X5_sub.iloc[tr_idx], y[tr_idx],
        eval_set=[(X5_sub.iloc[va_idx], y[va_idx])], eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    oof_5_sub[va_idx] = model.predict_proba(X5_sub.iloc[va_idx])[:, 1]
auc_5_sub = roc_auc_score(y, oof_5_sub)
print(f"LGBM(5 real features) on complete-case subset OOF AUC = {auc_5_sub:.5f}")

# Logistic regression with explicit pairwise interaction terms (not just squares)
Xarr = X5_sub.values
oof_interact = np.zeros(len(sub))
for tr, va in skf2.split(Xarr, y):
    scaler = StandardScaler().fit(Xarr[tr])
    Xtr_s, Xva_s = scaler.transform(Xarr[tr]), scaler.transform(Xarr[va])
    pair_idx = list(combinations(range(Xtr_s.shape[1]), 2))
    Xtr_pairs = np.hstack([Xtr_s[:, i] * Xtr_s[:, j] for i, j in pair_idx]).reshape(len(Xtr_s), -1) \
        if pair_idx else np.zeros((len(Xtr_s), 0))
    # build properly
    Xtr_pairs = np.column_stack([Xtr_s[:, i] * Xtr_s[:, j] for i, j in pair_idx])
    Xva_pairs = np.column_stack([Xva_s[:, i] * Xva_s[:, j] for i, j in pair_idx])
    Xtr_full = np.hstack([Xtr_s, Xtr_s ** 2, Xtr_pairs])
    Xva_full = np.hstack([Xva_s, Xva_s ** 2, Xva_pairs])
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr_full, y[tr])
    oof_interact[va] = lr.predict_proba(Xva_full)[:, 1]
auc_interact = roc_auc_score(y, oof_interact)
print(f"Logistic(5 feat + squares + pairwise products) on complete-case subset OOF AUC = {auc_interact:.5f}")

print("\n" + "=" * 70)
print("SUMMARY on complete-case subset (same rows for all 4 numbers below):")
print("=" * 70)
print(f"  Logistic, linear only            : 0.92735  (from exp004, same subset)")
print(f"  Logistic, + squares               : 0.93990  (from exp004, same subset)")
print(f"  Logistic, + squares + pairwise     : {auc_interact:.5f}")
print(f"  LGBM, 5 real features only         : {auc_5_sub:.5f}")
print(f"  LGBM, all 12 features              : {auc_12_sub:.5f}")
print(f"\nFor reference: LGBM(5 feat) on FULL 691k rows (native NaN)      : {auc_5feat_full:.5f}")
print(f"                LGBM(12 feat) on FULL 691k rows (exp001)         : 0.96384")
