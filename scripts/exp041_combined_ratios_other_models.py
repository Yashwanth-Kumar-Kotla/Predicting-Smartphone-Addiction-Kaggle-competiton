"""
Experiment 041 — Does exp040's super-additive combined-ratio gain
(+0.00051 on tuned XGBoost) generalize to LightGBM and CatBoost, the same
way entertainment_ratio alone did (exp035)?
"""
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

train, _ = load_data()
y = train[TARGET].values
train["entertainment_ratio"] = (train["social_media_hours"] + train["gaming_hours"]) / train["daily_screen_time_hours"]
train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
feat_cols = NUM_COLS + ["entertainment_ratio", "workstudy_ratio"]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("=" * 70)
print("LightGBM + both ratios")
print("=" * 70)
lgbm_params = dict(
    n_estimators=5000, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
    random_state=SEED, verbosity=-1,
)
oof_lgbm = np.zeros(len(train))
fold_aucs = []
t0 = time.time()
for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
    Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    model = LGBMClassifier(**lgbm_params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    pred = model.predict_proba(Xva)[:, 1]
    oof_lgbm[va_idx] = pred
    fold_aucs.append(roc_auc_score(yva, pred))
auc_lgbm = roc_auc_score(y, oof_lgbm)
print(f"OOF AUC: {auc_lgbm:.5f}  fold_std: {np.std(fold_aucs):.6f}  runtime: {time.time()-t0:.1f}s")
print(f"vs exp001 LightGBM baseline (0.96384): delta = {auc_lgbm - 0.96384:+.5f}")
np.save("artifacts/oof_exp041_lgbm_combined_ratios.npy", oof_lgbm)

print(f"\n{'=' * 70}\nCatBoost (tuned) + both ratios\n{'=' * 70}")
with open("artifacts/exp023_best_params.json") as f:
    catboost_tuned = json.load(f)
cb_params = dict(catboost_tuned, iterations=10000, loss_function="Logloss",
                  eval_metric="AUC", random_seed=SEED, early_stopping_rounds=150,
                  verbose=False, task_type="CPU")
oof_cb = np.zeros(len(train))
fold_aucs_cb = []
t0 = time.time()
for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
    Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    model = CatBoostClassifier(**cb_params)
    model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    pred = model.predict_proba(Xva)[:, 1]
    oof_cb[va_idx] = pred
    fold_aucs_cb.append(roc_auc_score(yva, pred))
auc_cb = roc_auc_score(y, oof_cb)
print(f"OOF AUC: {auc_cb:.5f}  fold_std: {np.std(fold_aucs_cb):.6f}  runtime: {time.time()-t0:.1f}s")
print(f"vs exp023 tuned CatBoost baseline (0.96401): delta = {auc_cb - 0.96401:+.5f}")
np.save("artifacts/oof_exp041_catboost_combined_ratios.npy", oof_cb)

log_experiment({
    "exp_id": "exp041",
    "model": "LightGBM + CatBoost, both + entertainment_ratio + workstudy_ratio",
    "features": "9 numeric + both ratios",
    "preprocessing": "none (native NaN handling)",
    "hyperparams": "lgbm: exp001 untuned; catboost: exp023 tuned, iterations 8000->10000",
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"lgbm={auc_lgbm:.5f} catboost={auc_cb:.5f}",
    "cv_std": f"lgbm={np.std(fold_aucs):.6f} catboost={np.std(fold_aucs_cb):.6f}",
    "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": f"{time.time()-t0:.1f}",
    "notes": "checking whether exp040's super-additive XGBoost gain (+0.00051) generalizes across tree families",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
