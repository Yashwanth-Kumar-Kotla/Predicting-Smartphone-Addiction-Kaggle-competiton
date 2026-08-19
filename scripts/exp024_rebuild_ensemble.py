"""
Experiment 024 — Rebuild the ensemble with the tuned XGBoost (exp022,
0.96473) and tuned CatBoost (exp023, 0.96401), keeping LightGBM untuned
(exp001, 0.96384) for now, and re-optimize blend weights. Cheap (uses
saved OOF arrays, no retraining) -- answers whether ~5.3hrs of HPO
compute actually moved the number that matters (the ensemble), before
deciding whether a 3rd HPO pass (LightGBM) is worth the same investment.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp001_lgbm_baseline.npy")
oof_catboost_v2 = np.load("artifacts/oof_exp023_catboost_tuned.npy")
oof_xgboost_v2 = np.load("artifacts/oof_exp022_xgboost_tuned_v2.npy")

names = ["lgbm", "catboost_v2", "xgboost_v2"]
oofs = [oof_lgbm, oof_catboost_v2, oof_xgboost_v2]
individual_aucs = [roc_auc_score(y, o) for o in oofs]

print("Individual OOF AUCs (new ensemble members):")
for n, a in zip(names, individual_aucs):
    print(f"  {n:15s} {a:.5f}")

corr = np.corrcoef(oofs)
print("\nPairwise correlation:")
print(f"{'':15s} " + " ".join(f"{n:>13s}" for n in names))
for i, n in enumerate(names):
    print(f"{n:15s} " + " ".join(f"{corr[i, j]:13.5f}" for j in range(len(names))))


def neg_auc(weights):
    w = np.abs(weights)
    w = w / w.sum()
    blend = sum(wi * oi for wi, oi in zip(w, oofs))
    return -roc_auc_score(y, blend)


x0 = np.ones(len(oofs)) / len(oofs)
res = minimize(neg_auc, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
best_w = np.abs(res.x)
best_w = best_w / best_w.sum()
best_blend = sum(wi * oi for wi, oi in zip(best_w, oofs))
best_auc = roc_auc_score(y, best_blend)

print(f"\nOOF-optimized weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, best_w)))
print(f"New ensemble OOF AUC: {best_auc:.5f}")
print(f"vs exp013 original ensemble (0.96476): delta = {best_auc - 0.96476:+.5f}")
print(f"vs exp022 tuned XGBoost alone (0.96473): delta = {best_auc - 0.96473:+.5f}")

np.save("artifacts/oof_exp024_ensemble_v2.npy", best_blend)
import json
with open("artifacts/exp024_ensemble_weights.json", "w") as f:
    json.dump(dict(zip(names, best_w.tolist())), f, indent=2)
