"""
Experiment 046 — entertainment_ratio = social_media_hours/screen_time +
gaming_hours/screen_time (summed numerator over the same denominator).
Test whether DECOMPOSING it into the 2 separate component ratios (instead
of the combined sum) reveals additional structure a tree can't get from
the combined version alone -- same productive "ratio, not sum" pattern
that worked for entertainment_ratio/workstudy_ratio, one more natural
extension before considering this feature family exhausted.
"""
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

train, _ = load_data()
y = train[TARGET].values

with open("artifacts/exp022_best_params.json") as f:
    xgb_tuned = json.load(f)
params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
              early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

train["social_ratio"] = train["social_media_hours"] / train["daily_screen_time_hours"]
train["gaming_ratio"] = train["gaming_hours"] / train["daily_screen_time_hours"]
train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]

# Variant A: decomposed (social_ratio + gaming_ratio) instead of combined entertainment_ratio
feat_a = NUM_COLS + ["social_ratio", "gaming_ratio", "workstudy_ratio"]
# Variant B: decomposed ratios ADDED alongside the combined entertainment_ratio
train["entertainment_ratio"] = train["social_ratio"] + train["gaming_ratio"]
feat_b = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def run(feat_cols, name):
    oof = np.zeros(len(train))
    fold_aucs = []
    t0 = time.time()
    for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
        Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred = model.predict_proba(Xva)[:, 1]
        oof[va_idx] = pred
        fold_aucs.append(roc_auc_score(yva, pred))
    auc = roc_auc_score(y, oof)
    print(f"{name}: OOF AUC={auc:.5f}  fold_std={np.std(fold_aucs):.6f}  runtime={time.time()-t0:.1f}s  n_features={len(feat_cols)}")
    return oof, auc


oof_a, auc_a = run(feat_a, "Variant A (decomposed only, no combined)")
oof_b, auc_b = run(feat_b, "Variant B (decomposed + combined)")

print(f"\nvs exp040 combined ratio only (0.96524):")
print(f"  Variant A delta: {auc_a - 0.96524:+.5f}")
print(f"  Variant B delta: {auc_b - 0.96524:+.5f}")

np.save("artifacts/oof_exp046_variantA.npy", oof_a)
np.save("artifacts/oof_exp046_variantB.npy", oof_b)

log_experiment({
    "exp_id": "exp046",
    "model": "XGBoost (tuned), decomposed entertainment_ratio variants",
    "features": "Variant A: 9 numeric + social_ratio + gaming_ratio + workstudy_ratio (no combined); Variant B: adds combined entertainment_ratio too",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}, same split as exp040",
    "cv_mean": f"A={auc_a:.5f} B={auc_b:.5f}",
    "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": "~1000",
    "notes": "does decomposing entertainment_ratio into social/gaming components add value beyond the combined ratio?",
    "conclusion": "TBD",
})
