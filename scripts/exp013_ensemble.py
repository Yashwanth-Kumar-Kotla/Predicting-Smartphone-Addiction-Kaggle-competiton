"""
Experiment 013 — Phase 7 (OOF analysis) + Phase 8 (ensembling).

Two feature-engineering attempts (exp010) and one HPO attempt (exp011/012)
both failed to beat exp008's untuned XGBoost baseline (0.96461). Before
sinking more budget into either, check whether our three existing diverse
model-family baselines (LightGBM exp001, CatBoost exp007, XGBoost exp008)
have low enough OOF prediction correlation to make ensembling worthwhile,
and find OOF-optimal blend weights (not equal-weight guessing).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp001_lgbm_baseline.npy")
oof_catboost = np.load("artifacts/oof_exp007_catboost_baseline.npy")
oof_xgboost = np.load("artifacts/oof_exp008_xgboost_baseline.npy")

names = ["lgbm", "catboost", "xgboost"]
oofs = [oof_lgbm, oof_catboost, oof_xgboost]
individual_aucs = [roc_auc_score(y, o) for o in oofs]

print("Individual OOF AUCs:")
for n, a in zip(names, individual_aucs):
    print(f"  {n:10s} {a:.5f}")

print("\nPairwise correlation of OOF predictions:")
corr = np.corrcoef(oofs)
print(f"{'':10s} " + " ".join(f"{n:>10s}" for n in names))
for i, n in enumerate(names):
    print(f"{n:10s} " + " ".join(f"{corr[i, j]:10.5f}" for j in range(len(names))))

# --- Equal-weight blend ---
equal_blend = np.mean(oofs, axis=0)
equal_auc = roc_auc_score(y, equal_blend)
print(f"\nEqual-weight blend OOF AUC: {equal_auc:.5f}")

# --- Rank-averaging blend (robust to different probability calibration) ---
from scipy.stats import rankdata
ranks = [rankdata(o) / len(o) for o in oofs]
rank_blend = np.mean(ranks, axis=0)
rank_auc = roc_auc_score(y, rank_blend)
print(f"Rank-averaging blend OOF AUC: {rank_auc:.5f}")

# --- OOF-optimized linear weights (constrained to simplex: weights >=0, sum=1) ---
def neg_auc(weights):
    w = np.abs(weights)
    w = w / w.sum()
    blend = sum(wi * oi for wi, oi in zip(w, oofs))
    return -roc_auc_score(y, blend)

x0 = np.ones(len(oofs)) / len(oofs)
res = minimize(neg_auc, x0, method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
best_w = np.abs(res.x)
best_w = best_w / best_w.sum()
best_blend = sum(wi * oi for wi, oi in zip(best_w, oofs))
best_auc = roc_auc_score(y, best_blend)

print(f"\nOOF-optimized weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, best_w)))
print(f"OOF-optimized blend AUC: {best_auc:.5f}")

print(f"\nBest single model (xgboost, exp008): {individual_aucs[names.index('xgboost')]:.5f}")
print(f"Gain from optimized ensemble: {best_auc - max(individual_aucs):+.5f}")

np.save("artifacts/oof_exp013_ensemble.npy", best_blend)
import json
with open("artifacts/exp013_ensemble_weights.json", "w") as f:
    json.dump(dict(zip(names, best_w.tolist())), f, indent=2)
