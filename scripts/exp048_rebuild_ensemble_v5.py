"""
Experiment 048 — Rebuild the ensemble with the final feature set (9 raw +
entertainment_ratio + social_ratio + gaming_ratio + workstudy_ratio)
across all 3 models (exp046 Variant B XGBoost 0.96538, exp047 LightGBM
0.96440 + CatBoost 0.96452).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp047_lgbm_final.npy")
oof_catboost = np.load("artifacts/oof_exp047_catboost_final.npy")
oof_xgboost = np.load("artifacts/oof_exp046_variantB.npy")

names = ["lgbm", "catboost", "xgboost"]
oofs = [oof_lgbm, oof_catboost, oof_xgboost]
individual_aucs = [roc_auc_score(y, o) for o in oofs]

print("Individual OOF AUCs (final feature set):")
for n, a in zip(names, individual_aucs):
    print(f"  {n:10s} {a:.5f}")


def neg_auc(weights):
    w = np.abs(weights)
    w = w / w.sum()
    blend = sum(wi * oi for wi, oi in zip(w, oofs))
    return -roc_auc_score(y, blend)


res = minimize(neg_auc, np.ones(len(oofs)) / len(oofs), method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
best_w = np.abs(res.x)
best_w = best_w / best_w.sum()
best_blend = sum(wi * oi for wi, oi in zip(best_w, oofs))
best_auc = roc_auc_score(y, best_blend)

print(f"\nOOF-optimized weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, best_w)))
print(f"New ensemble OOF AUC: {best_auc:.5f}")
print(f"vs exp042 ensemble (2 ratios only, 0.96542): delta = {best_auc - 0.96542:+.5f}")

np.save("artifacts/oof_exp048_ensemble_v5.npy", best_blend)
import json
with open("artifacts/exp048_ensemble_weights.json", "w") as f:
    json.dump(dict(zip(names, best_w.tolist())), f, indent=2)

from common import log_experiment
log_experiment({
    "exp_id": "exp048",
    "model": "Ensemble (LGBM+CatBoost+XGBoost, final feature set)",
    "features": "9 numeric + entertainment_ratio + social_ratio + gaming_ratio + workstudy_ratio",
    "preprocessing": "none",
    "hyperparams": "Nelder-Mead weight optimization on OOF",
    "cv_strategy": "n/a (uses existing 5-fold OOF arrays)",
    "cv_mean": f"{best_auc:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": "~2",
    "notes": "rebuild ensemble with final feature set (both ratios + decomposed components)",
    "conclusion": "TBD",
})
