"""
Experiment 055 — Ensemble using the HONEST (nested) OpenFE OOF for
XGBoost (exp054, 0.96604) combined with LightGBM/CatBoost's existing OOF
(exp047, no OpenFE features -- avoids repeating the expensive nested
procedure 2 more times). Using exp054's leakage-safe OOF for blend-weight
estimation (not exp051's leaky version) keeps the whole ensemble estimate
honest.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp047_lgbm_final.npy")
oof_catboost = np.load("artifacts/oof_exp047_catboost_final.npy")
oof_xgboost = np.load("artifacts/oof_exp054_nested_openfe.npy")

names = ["lgbm", "catboost", "xgboost"]
oofs = [oof_lgbm, oof_catboost, oof_xgboost]
individual_aucs = [roc_auc_score(y, o) for o in oofs]

print("Individual OOF AUCs:")
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
print(f"vs exp048 ensemble (no OpenFE features, 0.96553): delta = {best_auc - 0.96553:+.5f}")

np.save("artifacts/oof_exp055_ensemble_mixed.npy", best_blend)
import json
with open("artifacts/exp055_ensemble_weights.json", "w") as f:
    json.dump(dict(zip(names, best_w.tolist())), f, indent=2)

from common import log_experiment
log_experiment({
    "exp_id": "exp055",
    "model": "Ensemble (LGBM+CatBoost no-OpenFE, XGBoost with honest nested-OpenFE)",
    "features": "lgbm/catboost: current best (13); xgboost: current best (13) + nested OpenFE top 20",
    "preprocessing": "none",
    "hyperparams": "Nelder-Mead weight optimization on OOF",
    "cv_strategy": "n/a (uses existing OOF arrays, xgboost's OOF is leakage-safe nested)",
    "cv_mean": f"{best_auc:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": "~2",
    "notes": "mixed ensemble using honest nested-OpenFE OOF for XGBoost only",
    "conclusion": "TBD",
})
