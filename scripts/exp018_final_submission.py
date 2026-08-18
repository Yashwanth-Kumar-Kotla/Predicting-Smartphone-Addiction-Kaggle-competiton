"""
Experiment 018 — Phase 14: build the first real submission.

We have 17 CV experiments and a validated ensemble (exp013, OOF AUC
0.96476: xgboost=0.663, lgbm=0.168, catboost=0.169) but have never
actually produced a submission or gotten a public LB read. That signal
matters (Phase 10) -- it tells us whether our CV is trustworthy before we
invest more compute chasing further CV gains.

Refit each of the 3 models on 100% of training data using the SAME
hyperparameters validated in exp001/exp007/exp008, with n_estimators fixed
to the (rounded) mean best_iteration observed across that model's 5 CV
folds -- no early stopping possible without a held-out set, so we use the
CV-observed convergence point instead of guessing.
  LightGBM (exp001 folds: 3093,3256,3006,3701,3308) -> mean 3273
  CatBoost (exp007 folds: 4992,4997,4998,4995,4979) -> mean 4992 (never
    truly converged in CV, hit the 5000 cap every fold -- flagged as a
    future improvement, NOT changed here to keep this submission matched
    to what CV actually validated)
  XGBoost  (exp008 folds: 4064,3992,4160,4355,4497) -> mean 4214

Blend with exp013's OOF-optimized weights, write submission.csv, and run
Phase 15 sanity checks before declaring it done.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, ID_COL, SEED

train, test = load_data()
Xtr, ytr = train[NUM_COLS], train[TARGET].values
Xte = test[NUM_COLS]

print(f"train shape: {Xtr.shape}  test shape: {Xte.shape}")

# --- LightGBM ---
lgbm_params = dict(
    n_estimators=3273, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
    random_state=SEED, verbosity=-1,
)
print("\nTraining LightGBM on full data...")
lgbm = LGBMClassifier(**lgbm_params)
lgbm.fit(Xtr, ytr)
pred_lgbm = lgbm.predict_proba(Xte)[:, 1]
print("done.")

# --- CatBoost ---
catboost_params = dict(
    iterations=4992, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="Logloss", random_seed=SEED, verbose=False, task_type="CPU",
)
print("Training CatBoost on full data...")
catboost = CatBoostClassifier(**catboost_params)
catboost.fit(Xtr, ytr)
pred_catboost = catboost.predict_proba(Xte)[:, 1]
print("done.")

# --- XGBoost ---
xgb_params = dict(
    n_estimators=4214, learning_rate=0.03, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", random_state=SEED, n_jobs=-1,
)
print("Training XGBoost on full data...")
xgb = XGBClassifier(**xgb_params)
xgb.fit(Xtr, ytr)
pred_xgb = xgb.predict_proba(Xte)[:, 1]
print("done.")

# --- Blend with exp013 OOF-optimized weights ---
import json
with open("artifacts/exp013_ensemble_weights.json") as f:
    weights = json.load(f)
print(f"\nBlend weights: {weights}")

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
assert list(sample_sub.columns) == ["id", "addicted_label"], "unexpected sample_submission columns"

submission = pd.DataFrame({"id": test[ID_COL].values, "addicted_label": final_pred})
submission.to_csv("submission.csv", index=False)

print("\nAll Phase 15 sanity checks passed.")
print(f"submission.csv written: {len(submission)} rows")
print(submission.head())
print(f"\nprediction stats: min={final_pred.min():.5f} max={final_pred.max():.5f} mean={final_pred.mean():.5f}")
print(f"(train target mean for reference: {ytr.mean():.5f})")
