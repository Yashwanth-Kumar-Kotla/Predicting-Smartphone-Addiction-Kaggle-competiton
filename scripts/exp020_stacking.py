"""
Experiment 020 — Phase 8, proper stacking (not just weighted averaging).

exp016/017 showed the logistic/MLP models are real decorrelated signal
(corr 0.884/0.920 with the tree ensemble) but too weak individually for a
SIMPLE weighted blend to use them (optimizer always picked weight=0). A
nonlinear meta-learner can potentially do better than a single global
blend weight -- e.g. trust the diverse models more in specific regions of
feature space rather than everywhere or nowhere.

Uses 5 base OOF prediction arrays as meta-features:
  lgbm (exp001), catboost (exp019, the properly-converged version),
  xgboost (exp008), logistic (exp016), mlp (exp017)
Evaluated with a SECOND-LEVEL 5-fold CV on top of the (already honest) OOF
predictions, to get an honest estimate of how the stacker itself
generalizes -- not just fit it on all OOF data and read training AUC.

Tests both a linear (logistic) meta-learner and a small nonlinear
(GradientBoosting) meta-learner, plus a tree-only stack as a control to
isolate whether the diversity models add anything even via stacking.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from common import load_data, TARGET, N_FOLDS, SEED

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp001_lgbm_baseline.npy")
oof_catboost = np.load("artifacts/oof_exp019_catboost_more_iters.npy")
oof_xgboost = np.load("artifacts/oof_exp008_xgboost_baseline.npy")
oof_logistic = np.load("artifacts/oof_exp016_logistic_diversity.npy")
oof_mlp = np.load("artifacts/oof_exp017_mlp_diversity.npy")

all5 = np.column_stack([oof_lgbm, oof_catboost, oof_xgboost, oof_logistic, oof_mlp])
tree3 = np.column_stack([oof_lgbm, oof_catboost, oof_xgboost])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def eval_stacker(X, model_fn, name):
    meta_oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        model = model_fn()
        model.fit(X[tr_idx], y[tr_idx])
        meta_oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
    auc = roc_auc_score(y, meta_oof)
    print(f"{name:45s} AUC={auc:.5f}")
    return auc, meta_oof


print("=" * 70)
print("Baseline for reference: exp013 simple weighted blend (3 trees) = 0.96476")
print("=" * 70)

auc_tree3_lr, _ = eval_stacker(tree3, lambda: LogisticRegression(), "Stack(3 trees) + Logistic meta")
auc_all5_lr, _ = eval_stacker(all5, lambda: LogisticRegression(), "Stack(3 trees+logistic+mlp) + Logistic meta")
auc_all5_gbm, oof_gbm_stack = eval_stacker(
    all5, lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=SEED),
    "Stack(3 trees+logistic+mlp) + GBM meta (nonlinear)"
)
auc_tree3_gbm, _ = eval_stacker(
    tree3, lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=SEED),
    "Stack(3 trees only) + GBM meta (control)"
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"exp013 simple weighted blend (3 trees, linear weights): 0.96476")
print(f"Stack(3 trees) + Logistic meta:                          {auc_tree3_lr:.5f}")
print(f"Stack(5 models) + Logistic meta:                         {auc_all5_lr:.5f}")
print(f"Stack(3 trees) + GBM meta (nonlinear, control):          {auc_tree3_gbm:.5f}")
print(f"Stack(5 models) + GBM meta (nonlinear):                  {auc_all5_gbm:.5f}")
best_name = max(
    [("tree3_lr", auc_tree3_lr), ("all5_lr", auc_all5_lr), ("tree3_gbm", auc_tree3_gbm), ("all5_gbm", auc_all5_gbm)],
    key=lambda x: x[1]
)
print(f"\nBest stacker: {best_name[0]} = {best_name[1]:.5f}  (delta vs exp013: {best_name[1]-0.96476:+.5f})")

if best_name[0] == "all5_gbm":
    np.save("artifacts/oof_exp020_stack.npy", oof_gbm_stack)
    print("Saved best stacker's OOF predictions to artifacts/oof_exp020_stack.npy")
