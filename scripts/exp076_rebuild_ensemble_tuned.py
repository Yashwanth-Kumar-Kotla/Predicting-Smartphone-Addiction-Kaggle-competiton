"""
Experiment 076 — Rebuild ensemble with exp075's tuned XGBoost (0.96732)
replacing the default-params version (0.96721), combined with existing
LightGBM/CatBoost consensus OOF (exp072).
"""
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp072_lgbm_consensus.npy")
oof_catboost = np.load("artifacts/oof_exp072_catboost_consensus.npy")
oof_xgboost = np.load("artifacts/oof_exp075_xgb_hpo_consensus.npy")

names = ["lgbm", "catboost", "xgboost"]
oofs = [oof_lgbm, oof_catboost, oof_xgboost]
individual_aucs = [roc_auc_score(y, o) for o in oofs]

print("Individual OOF AUCs (consensus features, XGBoost tuned):")
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
print(f"vs exp073 ensemble (untuned XGBoost, 0.96734): delta = {best_auc - 0.96734:+.5f}")

np.save("artifacts/oof_exp076_ensemble_tuned.npy", best_blend)
with open("artifacts/exp076_ensemble_weights.json", "w") as f:
    json.dump(dict(zip(names, best_w.tolist())), f, indent=2)

from common import log_experiment
log_experiment({
    "exp_id": "exp076",
    "model": "Ensemble (LGBM+CatBoost+tuned-XGBoost, consensus-9 features)",
    "features": "current best (13) + 9 consensus OpenFE features, all 3 models",
    "preprocessing": "none",
    "hyperparams": "Nelder-Mead weight optimization on OOF",
    "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV throughout",
    "cv_mean": f"{best_auc:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": "~2",
    "notes": "ensemble with exp075's tuned XGBoost replacing default params",
    "conclusion": "TBD",
})
