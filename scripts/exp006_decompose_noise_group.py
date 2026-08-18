"""
Experiment 006 — exp005 showed adding all 7 "weak" features (sleep_hours,
notifications_per_day, app_opens_per_day, age, gender, stress_level,
academic_work_impact) to the 5 "real" features lifts LGBM AUC by +0.0154
on a fixed complete-case subset, despite each having near-noise univariate
AUC individually. Decompose: is that gain from the 4 weak numeric features,
the 3 categoricals, or both jointly? Also directly answer the original
question -- does dropping ONLY gender cost anything in a CV ablation?

All runs use the SAME complete-case subset (rows with no missing in the 5
"real" features) as exp005, for a clean apples-to-apples comparison.
"""
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from common import load_data, FEATURE_COLS, CAT_COLS, TARGET, N_FOLDS, SEED

REAL5 = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
         "work_study_hours", "gaming_hours"]
WEAK_NUM = ["sleep_hours", "notifications_per_day", "app_opens_per_day", "age"]
WEAK_CAT = ["gender", "stress_level", "academic_work_impact"]

train, _ = load_data()
mask = train[REAL5].notna().all(axis=1)
sub = train.loc[mask].reset_index(drop=True)
y = sub[TARGET].values
print(f"subset size: {len(sub)} rows")

lgb_params = dict(
    n_estimators=5000, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, verbosity=-1,
)

variants = {
    "5real": REAL5,
    "5real+4weaknum": REAL5 + WEAK_NUM,
    "5real+3weakcat": REAL5 + WEAK_CAT,
    "all12_minus_gender": [c for c in FEATURE_COLS if c != "gender"],
    "all12": FEATURE_COLS,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results = {}
for name, cols in variants.items():
    cat_cols_here = [c for c in cols if c in CAT_COLS]
    X = sub[cols]
    oof = np.zeros(len(sub))
    t0 = time.time()
    for tr_idx, va_idx in skf.split(X, y):
        model = LGBMClassifier(**lgb_params)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])], eval_metric="auc",
            categorical_feature=cat_cols_here if cat_cols_here else "auto",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    auc = roc_auc_score(y, oof)
    results[name] = auc
    print(f"{name:25s} n_features={len(cols):2d}  AUC={auc:.5f}  ({time.time()-t0:.0f}s)")

print("\n" + "=" * 70)
print("SUMMARY (all on identical complete-case subset)")
print("=" * 70)
for name, auc in results.items():
    print(f"  {name:25s} {auc:.5f}")
print(f"\ngain from 4 weak numeric alone : {results['5real+4weaknum'] - results['5real']:+.5f}")
print(f"gain from 3 weak categorical alone: {results['5real+3weakcat'] - results['5real']:+.5f}")
print(f"gain from ALL 7 weak together (exp005 all12): 0.96983 - 0.95439 = +0.01544")
print(f"cost of dropping gender specifically: {results['all12'] - results['all12_minus_gender']:+.5f}")
