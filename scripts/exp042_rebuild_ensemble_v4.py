"""
Experiment 042 — Rebuild the ensemble with both ratios (entertainment_ratio,
workstudy_ratio) added to all 3 tree models (exp040 XGBoost 0.96524,
exp041 LightGBM 0.96434 + CatBoost 0.96444), re-optimize blend weights.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp041_lgbm_combined_ratios.npy")
oof_catboost = np.load("artifacts/oof_exp041_catboost_combined_ratios.npy")
oof_xgboost = np.load("artifacts/oof_exp040_combined_ratios.npy")

names = ["lgbm", "catboost", "xgboost"]
oofs = [oof_lgbm, oof_catboost, oof_xgboost]
individual_aucs = [roc_auc_score(y, o) for o in oofs]

print("Individual OOF AUCs (with both ratios):")
for n, a in zip(names, individual_aucs):
    print(f"  {n:10s} {a:.5f}")

corr = np.corrcoef(oofs)
print("\nPairwise correlation:")
for i, n in enumerate(names):
    print(f"  {n:10s} " + " ".join(f"{corr[i, j]:.5f}" for j in range(len(names))))


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
print(f"vs exp036 ensemble (entertainment_ratio only, 0.96511): delta = {best_auc - 0.96511:+.5f}")
print(f"vs exp024 ensemble (no ratios, 0.96494): delta = {best_auc - 0.96494:+.5f}")

np.save("artifacts/oof_exp042_ensemble_v4.npy", best_blend)
import json
with open("artifacts/exp042_ensemble_weights.json", "w") as f:
    json.dump(dict(zip(names, best_w.tolist())), f, indent=2)

from common import log_experiment
log_experiment({
    "exp_id": "exp042",
    "model": "Ensemble (LGBM+CatBoost+XGBoost, all + both ratios)",
    "features": "9 numeric + entertainment_ratio + workstudy_ratio",
    "preprocessing": "none",
    "hyperparams": "Nelder-Mead weight optimization on OOF",
    "cv_strategy": "n/a (uses existing 5-fold OOF arrays)",
    "cv_mean": f"{best_auc:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": "~2",
    "notes": "rebuild ensemble with both ratios added to all 3 models",
    "conclusion": "TBD",
})
