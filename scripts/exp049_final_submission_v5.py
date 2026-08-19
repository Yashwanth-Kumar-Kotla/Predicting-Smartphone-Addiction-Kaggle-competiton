"""
Experiment 049 — Updated multi-seed submission with the final feature set
(entertainment_ratio, social_ratio, gaming_ratio, workstudy_ratio) added
to all 3 models (exp046/047/048). New ensemble CV = 0.96553 (+0.00011
over the previous best submission's basis, exp042's 0.96542).

Same proven approach as exp025/037/043: 3 seeds, small 5% stratified
holdout per seed for early-stopping determination, blend with exp048's
re-optimized weights (lgbm=0.082, catboost=0.240, xgboost=0.678).
"""
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, ID_COL, SEED

train, test = load_data()
for df in (train, test):
    df["social_ratio"] = df["social_media_hours"] / df["daily_screen_time_hours"]
    df["gaming_ratio"] = df["gaming_hours"] / df["daily_screen_time_hours"]
    df["entertainment_ratio"] = df["social_ratio"] + df["gaming_ratio"]
    df["workstudy_ratio"] = df["work_study_hours"] / df["daily_screen_time_hours"]
feat_cols = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]

Xfull, yfull = train[feat_cols], train[TARGET].values
Xte = test[feat_cols]

with open("artifacts/exp022_best_params.json") as f:
    xgb_tuned = json.load(f)
with open("artifacts/exp023_best_params.json") as f:
    catboost_tuned = json.load(f)

SEEDS = [42, 202, 2026]
preds = {"lgbm": [], "catboost": [], "xgboost": []}

t0 = time.time()
for seed in SEEDS:
    print(f"\n--- seed {seed} ---")
    Xtr, Xval, ytr, yval = train_test_split(
        Xfull, yfull, test_size=0.05, stratify=yfull, random_state=seed
    )

    print("LightGBM...")
    lgbm_params = dict(
        n_estimators=6000, learning_rate=0.03, num_leaves=63, max_depth=-1,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
        random_state=seed, verbosity=-1,
    )
    m = LGBMClassifier(**lgbm_params)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="auc",
          callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    preds["lgbm"].append(m.predict_proba(Xte)[:, 1])
    print(f"  best_iter={m.best_iteration_}")

    print("CatBoost...")
    cb_params = dict(catboost_tuned, iterations=10000, loss_function="Logloss",
                      eval_metric="AUC", random_seed=seed, early_stopping_rounds=150,
                      verbose=False, task_type="CPU")
    m = CatBoostClassifier(**cb_params)
    m.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
    preds["catboost"].append(m.predict_proba(Xte)[:, 1])
    print(f"  best_iter={m.get_best_iteration()}")

    print("XGBoost...")
    xgb_params = dict(xgb_tuned, n_estimators=7000, tree_method="hist",
                       eval_metric="auc", early_stopping_rounds=100,
                       random_state=seed, n_jobs=-1)
    m = XGBClassifier(**xgb_params)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    preds["xgboost"].append(m.predict_proba(Xte)[:, 1])
    print(f"  best_iter={m.best_iteration}")

    print(f"seed {seed} done, elapsed {time.time()-t0:.0f}s")

pred_lgbm = np.mean(preds["lgbm"], axis=0)
pred_catboost = np.mean(preds["catboost"], axis=0)
pred_xgb = np.mean(preds["xgboost"], axis=0)

print(f"\nTotal runtime: {time.time()-t0:.1f}s")

np.save("artifacts/pred_test_exp049_lgbm.npy", pred_lgbm)
np.save("artifacts/pred_test_exp049_catboost.npy", pred_catboost)
np.save("artifacts/pred_test_exp049_xgboost.npy", pred_xgb)

with open("artifacts/exp048_ensemble_weights.json") as f:
    weights = json.load(f)
print(f"Blend weights (from exp048): {weights}")

final_pred = (
    weights["lgbm"] * pred_lgbm
    + weights["catboost"] * pred_catboost
    + weights["xgboost"] * pred_xgb
)
final_pred = np.clip(final_pred, 0.0, 1.0)

sample_sub = pd.read_csv("playground-series-s6e8/sample_submission.csv")
assert len(final_pred) == len(test)
assert (test[ID_COL].values == sample_sub[ID_COL].values).all()
assert np.isfinite(final_pred).all()
assert (final_pred >= 0).all() and (final_pred <= 1).all()

submission = pd.DataFrame({"id": test[ID_COL].values, "addicted_label": final_pred})
submission.to_csv("submission_v5.csv", index=False)

old_sub = pd.read_csv("submission_v4.csv")
diff = (submission["addicted_label"] - old_sub["addicted_label"]).abs()
print(f"\nAll Phase 15 sanity checks passed.")
print(f"submission_v5.csv written: {len(submission)} rows")
print(f"mean |change| vs submission_v4.csv (prev best): {diff.mean():.5f}")
print(f"correlation with submission_v4.csv: {submission['addicted_label'].corr(old_sub['addicted_label']):.6f}")
print(f"\nprediction stats: min={final_pred.min():.5f} max={final_pred.max():.5f} mean={final_pred.mean():.5f}")
print(f"(train target mean for reference: {yfull.mean():.5f})")
