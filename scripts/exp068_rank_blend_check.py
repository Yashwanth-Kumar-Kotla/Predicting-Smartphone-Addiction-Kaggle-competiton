"""
Experiment 068 — A discussion thread suggested rank-based blending gives
a more stable boost than probability blending. Most other claims in that
thread contradict what we've rigorously found (target encoding on
nonexistent high-cardinality categoricals, interaction features already
shown null for trees, stacking already shown null in exp020), but this
one is cheap, untested, and not contradicted -- checking it directly on
our current best (top-40 OpenFE) ensemble.
"""
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common import load_data, TARGET

train, _ = load_data()
y = train[TARGET].values

oof_lgbm = np.load("artifacts/oof_exp063_nested_openfe_top40_lgbm.npy")
oof_catboost = np.load("artifacts/oof_exp064_nested_openfe_top40_catboost.npy")
oof_xgboost = np.load("artifacts/oof_exp061_nested_openfe_top40_xgb.npy")

names = ["lgbm", "catboost", "xgboost"]
oofs = [oof_lgbm, oof_catboost, oof_xgboost]

# --- Current best: probability-weighted blend (exp065) ---
with open("artifacts/exp065_ensemble_weights.json") as f:
    import json
    weights = json.load(f)
prob_blend = sum(weights[n] * o for n, o in zip(names, oofs))
prob_auc = roc_auc_score(y, prob_blend)
print(f"Current best (probability-weighted blend, exp065): {prob_auc:.5f}")

# --- Equal-weight rank blend ---
ranks = [rankdata(o) / len(o) for o in oofs]
equal_rank_blend = np.mean(ranks, axis=0)
equal_rank_auc = roc_auc_score(y, equal_rank_blend)
print(f"Equal-weight rank blend: {equal_rank_auc:.5f}")

# --- OOF-optimized rank blend ---
def neg_auc_rank(w):
    w = np.abs(w); w = w / w.sum()
    blend = sum(wi * ri for wi, ri in zip(w, ranks))
    return -roc_auc_score(y, blend)

res = minimize(neg_auc_rank, np.ones(3) / 3, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8})
best_w_rank = np.abs(res.x); best_w_rank = best_w_rank / best_w_rank.sum()
best_rank_blend = sum(wi * ri for wi, ri in zip(best_w_rank, ranks))
best_rank_auc = roc_auc_score(y, best_rank_blend)
print(f"OOF-optimized rank blend: {best_rank_auc:.5f}  weights=" + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, best_w_rank)))

print(f"\nDelta (optimized rank vs current probability blend): {best_rank_auc - prob_auc:+.5f}")
