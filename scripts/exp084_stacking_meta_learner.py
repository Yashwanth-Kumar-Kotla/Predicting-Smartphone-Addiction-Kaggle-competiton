"""
Experiment 084 — Per-model HPO is now exhausted (XGBoost +0.00011, LightGBM
+0.00045, CatBoost no headroom). The current ensemble (exp080, 0.96747) uses
a Nelder-Mead-optimized linear blend of the 3 OOF prediction arrays. Testing
whether a shallow non-linear stacker (small GBM trained ON the 3 OOF columns
as features) captures synergy the linear blend can't -- e.g. regions where
one model should be trusted more depending on another's prediction.

Meta-learner is evaluated with its own proper CV (5-fold on the OOF arrays)
to get an honest estimate, not fit-and-eval on the same data.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from common import load_data, TARGET, N_FOLDS, SEED

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp079_lgbm_hpo_consensus.npy")
oof_catboost = np.load("artifacts/oof_exp072_catboost_consensus.npy")
oof_xgboost = np.load("artifacts/oof_exp075_xgb_hpo_consensus.npy")

Xmeta = np.column_stack([oof_lgbm, oof_catboost, oof_xgboost])
baseline_auc = 0.96747  # exp080 Nelder-Mead linear blend

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 1. Logistic regression stacker (linear, but with intercept + free weights vs simplex)
oof_lr = np.zeros(len(y))
for tr_idx, va_idx in skf.split(Xmeta, y):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xmeta[tr_idx], y[tr_idx])
    oof_lr[va_idx] = lr.predict_proba(Xmeta[va_idx])[:, 1]
auc_lr = roc_auc_score(y, oof_lr)
print(f"Logistic regression stacker OOF AUC: {auc_lr:.5f}  (vs Nelder-Mead blend {baseline_auc:.5f}): delta={auc_lr-baseline_auc:+.5f}")

# 2. Shallow LightGBM stacker (non-linear, captures interactions between the 3 model outputs)
oof_gbm = np.zeros(len(y))
for tr_idx, va_idx in skf.split(Xmeta, y):
    gbm = LGBMClassifier(
        n_estimators=200, max_depth=3, num_leaves=7, learning_rate=0.03,
        min_child_samples=200, subsample=0.8, colsample_bytree=1.0,
        reg_alpha=1.0, reg_lambda=1.0, random_state=SEED, verbosity=-1,
    )
    gbm.fit(Xmeta[tr_idx], y[tr_idx])
    oof_gbm[va_idx] = gbm.predict_proba(Xmeta[va_idx])[:, 1]
auc_gbm = roc_auc_score(y, oof_gbm)
print(f"Shallow LightGBM stacker OOF AUC: {auc_gbm:.5f}  (vs Nelder-Mead blend {baseline_auc:.5f}): delta={auc_gbm-baseline_auc:+.5f}")

best_name, best_auc, best_oof = max(
    [("logistic_regression", auc_lr, oof_lr), ("shallow_lightgbm", auc_gbm, oof_gbm)],
    key=lambda t: t[1],
)
print(f"\nBest stacker: {best_name} ({best_auc:.5f})")

if best_auc > baseline_auc:
    np.save("artifacts/oof_exp084_best_stacker.npy", best_oof)
    print("Saved -- beats linear blend, worth deploying.")
else:
    print("Neither stacker beats the linear blend -- linear blend remains best.")

from common import log_experiment
log_experiment({
    "exp_id": "exp084",
    "model": "Stacking meta-learner comparison (logistic regression vs shallow LightGBM) on 3 base-model OOF predictions",
    "features": "3 OOF prediction columns (lgbm-tuned, catboost, xgboost-tuned)",
    "preprocessing": "none",
    "hyperparams": f"LR: max_iter=1000 default C; LightGBM: n_estimators=200 max_depth=3 num_leaves=7 lr=0.03 reg_alpha=1 reg_lambda=1",
    "cv_strategy": "StratifiedKFold n=5 seed=42 on OOF arrays (proper nested eval, not fit-and-eval-in-sample)",
    "cv_mean": f"LR={auc_lr:.5f} GBM={auc_gbm:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": "~5",
    "notes": "testing whether non-linear stacking beats exp080's Nelder-Mead linear blend (0.96747)",
    "conclusion": "TBD",
})
print("\nLogged exp084.")
