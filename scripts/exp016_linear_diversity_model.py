"""
Experiment 016 — Phase 8, targeted diversity attempt.

exp013 showed our 3 tree models correlate at 0.99+ on OOF predictions --
no room left to gain from ensembling more trees. exp010 showed pairwise
interaction terms are redundant FOR TREES (they can already build them via
splits) but never tested whether those same terms help a model that
structurally cannot form interactions on its own: regularized logistic
regression. If this model is even moderately accurate AND meaningfully
less correlated with the tree OOF predictions, it's a much better
ensemble contributor than another GBM would be.

Features: 9 raw numeric + 5 squared terms + 10 pairwise products of the
5 "real" features (exp003/009) = 24 total. Per-fold median imputation
(required -- sklearn LogisticRegression can't handle NaN) + standardization,
both fit on the train fold only.
"""
import time
from itertools import combinations
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

REAL5 = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
         "work_study_hours", "gaming_hours"]

train, _ = load_data()
y = train[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(train[NUM_COLS], y)):
    Xtr_raw = train[NUM_COLS].iloc[tr_idx].copy()
    Xva_raw = train[NUM_COLS].iloc[va_idx].copy()
    ytr, yva = y[tr_idx], y[va_idx]

    medians = Xtr_raw.median()
    Xtr_raw = Xtr_raw.fillna(medians)
    Xva_raw = Xva_raw.fillna(medians)

    def engineer(df):
        feats = {c: df[c].values for c in NUM_COLS}
        for c in REAL5:
            feats[f"{c}_sq"] = df[c].values ** 2
        for f1, f2 in combinations(REAL5, 2):
            feats[f"{f1}_x_{f2}"] = df[f1].values * df[f2].values
        return np.column_stack(list(feats.values()))

    Xtr = engineer(Xtr_raw)
    Xva = engineer(Xva_raw)

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xva = scaler.transform(Xva)

    model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.3,
                                C=1.0, max_iter=3000, random_state=SEED)
    model.fit(Xtr, ytr)
    pred = model.predict_proba(Xva)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(yva, pred)
    fold_aucs.append(auc)
    print(f"fold {fold}: AUC={auc:.5f}")

runtime = time.time() - t0
oof_auc = roc_auc_score(y, oof)
fold_std = float(np.std(fold_aucs))

print(f"\nOOF AUC: {oof_auc:.5f}")
print(f"fold mean: {np.mean(fold_aucs):.5f}  fold std: {fold_std:.6f}")
print(f"runtime: {runtime:.1f}s")

np.save("artifacts/oof_exp016_logistic_diversity.npy", oof)

# --- Correlation with existing tree ensemble + blend test ---
tree_oof = np.load("artifacts/oof_exp013_ensemble.npy")
corr = np.corrcoef(oof, tree_oof)[0, 1]
print(f"\nCorrelation with exp013 tree ensemble OOF: {corr:.5f}  (trees correlate 0.993-0.995 with each other)")

from scipy.optimize import minimize
def neg_auc(w):
    w = np.abs(w); w = w / w.sum()
    blend = w[0] * tree_oof + w[1] * oof
    return -roc_auc_score(y, blend)
res = minimize(neg_auc, [0.8, 0.2], method="Nelder-Mead")
best_w = np.abs(res.x); best_w = best_w / best_w.sum()
blend_auc = roc_auc_score(y, best_w[0] * tree_oof + best_w[1] * oof)
print(f"Best blend weights: tree_ensemble={best_w[0]:.3f}, logistic={best_w[1]:.3f}")
print(f"Blended OOF AUC: {blend_auc:.5f}  (vs tree-ensemble-alone 0.96476)")
print(f"Gain from adding logistic model to ensemble: {blend_auc - 0.96476:+.5f}")

log_experiment({
    "exp_id": "exp016",
    "model": "LogisticRegression (ElasticNet)",
    "features": "9 raw + 5 squared + 10 pairwise products of the 5 real features (24 total)",
    "preprocessing": "per-fold median imputation + StandardScaler, both fit on train fold only",
    "hyperparams": "penalty=elasticnet, solver=saga, l1_ratio=0.3, C=1.0, max_iter=3000",
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": f"diversity model for ensemble; corr with tree ensemble={corr:.5f}; blend AUC={blend_auc:.5f}",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
