"""
Experiment 021 — Multi-seed bagging for the final submission.

8 experiments (exp010-020) exhausted feature engineering, HPO, imputation,
diversity-model blending/stacking, and a CatBoost undertraining fix without
beating exp013's 0.96476 CV / the resulting 0.96606 public LB. Rather than
another speculative single-lever tweak, apply a safe, well-established
variance-reduction technique: retrain each of the 3 base models (LightGBM,
CatBoost-v2 from exp019, XGBoost) at 3 different seeds on 100% of the
training data, average WITHIN each model family, then blend ACROSS
families using exp013's OOF-optimized weights. This can't easily hurt
(averaging correlated-but-not-identical models reduces prediction
variance) and is standard practice for squeezing a final small, reliable
gain close to a deadline.

n_estimators per model/seed uses the mean best_iteration values already
validated in CV (same as exp018), scaled up ~10% as headroom since
different seeds can have slightly different optimal stopping points and
we have no per-seed validation set to early-stop against.
"""
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, ID_COL

train, test = load_data()
Xtr, ytr = train[NUM_COLS], train[TARGET].values
Xte = test[NUM_COLS]

SEEDS = [42, 202, 2026]

lgbm_base_params = dict(
    n_estimators=3600, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0, verbosity=-1,
)
catboost_base_params = dict(
    iterations=7500, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", verbose=False, task_type="CPU",
)
xgb_base_params = dict(
    n_estimators=4600, learning_rate=0.03, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", n_jobs=-1,
)

preds = {"lgbm": [], "catboost": [], "xgboost": []}
t0 = time.time()
for seed in SEEDS:
    print(f"\n--- seed {seed} ---")

    print("LightGBM...")
    m = LGBMClassifier(**lgbm_base_params, random_state=seed)
    m.fit(Xtr, ytr)
    preds["lgbm"].append(m.predict_proba(Xte)[:, 1])

    print("CatBoost...")
    m = CatBoostClassifier(**catboost_base_params, random_seed=seed)
    m.fit(Xtr, ytr)
    preds["catboost"].append(m.predict_proba(Xte)[:, 1])

    print("XGBoost...")
    m = XGBClassifier(**xgb_base_params, random_state=seed)
    m.fit(Xtr, ytr)
    preds["xgboost"].append(m.predict_proba(Xte)[:, 1])

    print(f"seed {seed} done, elapsed {time.time()-t0:.0f}s")

pred_lgbm = np.mean(preds["lgbm"], axis=0)
pred_catboost = np.mean(preds["catboost"], axis=0)
pred_xgb = np.mean(preds["xgboost"], axis=0)

print(f"\nTotal runtime: {time.time()-t0:.1f}s")

import json
with open("artifacts/exp013_ensemble_weights.json") as f:
    weights = json.load(f)
print(f"Blend weights (from exp013): {weights}")

final_pred = (
    weights["lgbm"] * pred_lgbm
    + weights["catboost"] * pred_catboost
    + weights["xgboost"] * pred_xgb
)

# --- Phase 15 sanity checks ---
sample_sub = pd.read_csv("playground-series-s6e8/sample_submission.csv")
assert len(final_pred) == len(test), f"length mismatch: {len(final_pred)} vs {len(test)}"
assert (test[ID_COL].values == sample_sub[ID_COL].values).all(), "id order mismatch vs sample_submission"
assert np.isfinite(final_pred).all(), "found NaN/inf in predictions"
assert (final_pred >= 0).all() and (final_pred <= 1).all(), "predictions out of [0,1] range"

submission = pd.DataFrame({"id": test[ID_COL].values, "addicted_label": final_pred})
submission.to_csv("submission_multiseed.csv", index=False)

# Compare against the single-seed submission already on disk
old_sub = pd.read_csv("submission.csv")
diff = (submission["addicted_label"] - old_sub["addicted_label"]).abs()
print(f"\nAll Phase 15 sanity checks passed.")
print(f"submission_multiseed.csv written: {len(submission)} rows")
print(f"mean |change| vs original submission.csv: {diff.mean():.5f}")
print(f"max |change| vs original submission.csv: {diff.max():.5f}")
print(f"correlation with original submission.csv: {submission['addicted_label'].corr(old_sub['addicted_label']):.6f}")
print(f"\nprediction stats: min={final_pred.min():.5f} max={final_pred.max():.5f} mean={final_pred.mean():.5f}")
print(f"(train target mean for reference: {ytr.mean():.5f})")
