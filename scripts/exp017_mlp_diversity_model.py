"""
Experiment 017 — Phase 8, second diversity attempt.

exp016 showed a linear model gets real decorrelation from the tree
ensemble (corr 0.884) but is too inaccurate (0.92451, 4pts below trees) to
add value in a blend. A small MLP can learn interactions/nonlinearities
itself (same reason trees don't need hand-crafted interaction terms,
exp010) without a linear model's imputation sensitivity, so it has a
better shot at BOTH staying reasonably accurate AND remaining
architecturally different enough to decorrelate.

Features: 9 raw numeric only (let the network find structure itself, same
philosophy as the tree baselines -- no hand-crafted terms). Per-fold
median imputation + StandardScaler, both fit on train fold only (required,
MLPClassifier can't handle NaN).
"""
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

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

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xva = scaler.transform(Xva_raw)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=2048,
        learning_rate_init=1e-3,
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=SEED,
    )
    model.fit(Xtr, ytr)
    pred = model.predict_proba(Xva)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(yva, pred)
    fold_aucs.append(auc)
    print(f"fold {fold}: AUC={auc:.5f}  n_iter={model.n_iter_}")

runtime = time.time() - t0
oof_auc = roc_auc_score(y, oof)
fold_std = float(np.std(fold_aucs))

print(f"\nOOF AUC: {oof_auc:.5f}")
print(f"fold mean: {np.mean(fold_aucs):.5f}  fold std: {fold_std:.6f}")
print(f"runtime: {runtime:.1f}s")

np.save("artifacts/oof_exp017_mlp_diversity.npy", oof)

tree_oof = np.load("artifacts/oof_exp013_ensemble.npy")
corr = np.corrcoef(oof, tree_oof)[0, 1]
print(f"\nCorrelation with exp013 tree ensemble OOF: {corr:.5f}")

from scipy.optimize import minimize
def neg_auc(w):
    w = np.abs(w); w = w / w.sum()
    blend = w[0] * tree_oof + w[1] * oof
    return -roc_auc_score(y, blend)
res = minimize(neg_auc, [0.8, 0.2], method="Nelder-Mead")
best_w = np.abs(res.x); best_w = best_w / best_w.sum()
blend_auc = roc_auc_score(y, best_w[0] * tree_oof + best_w[1] * oof)
print(f"Best blend weights: tree_ensemble={best_w[0]:.3f}, mlp={best_w[1]:.3f}")
print(f"Blended OOF AUC: {blend_auc:.5f}  (vs tree-ensemble-alone 0.96476)")
print(f"Gain from adding MLP to ensemble: {blend_auc - 0.96476:+.5f}")

log_experiment({
    "exp_id": "exp017",
    "model": "MLPClassifier (sklearn)",
    "features": "9 raw numeric only",
    "preprocessing": "per-fold median imputation + StandardScaler, both fit on train fold only",
    "hyperparams": "hidden=(128,64), relu, adam, alpha=1e-4, batch=2048, early_stopping=True",
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": f"diversity model attempt 2; corr with tree ensemble={corr:.5f}; blend AUC={blend_auc:.5f}",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
