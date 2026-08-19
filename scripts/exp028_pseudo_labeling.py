"""
Experiment 028 — Pseudo-labeling, the one genuinely untried lever flagged
after exhausting FE/HPO/ensembling/stacking/imputation/MLP-diversity.

Select test rows where the current best full-data ensemble (submission_v2,
CV 0.96494) predicts with high confidence (<0.02 or >0.98) as pseudo-
labeled training rows -- 145,510 of 296,302 test rows (49.1%) qualify.

LEAKAGE-SAFE DESIGN: the original 5-fold split (same seed=42) is kept
fixed for evaluation. Pseudo-labeled rows are added ONLY to each fold's
TRAINING portion, never to the validation portion -- we don't know their
true labels, so evaluating against them would be circular. This isolates
whether pseudo-labeling helps GENERALIZATION on real held-out data.

Uses tuned XGBoost (exp022 params) as the test model since it's the
strongest single model and cheapest to iterate on.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment

train, test = load_data()
X, y = train[NUM_COLS], train[TARGET].values

sub = pd.read_csv("submission_v2.csv")
confident_mask = (sub["addicted_label"] < 0.02) | (sub["addicted_label"] > 0.98)
pseudo_X = test.loc[confident_mask, NUM_COLS].reset_index(drop=True)
pseudo_y = (sub.loc[confident_mask, "addicted_label"] > 0.5).astype(int).values
print(f"Pseudo-labeled rows: {len(pseudo_X)} ({len(pseudo_X)/len(test)*100:.1f}% of test)")
print(f"  positive: {(pseudo_y==1).sum()}  negative: {(pseudo_y==0).sum()}")

import json
with open("artifacts/exp022_best_params.json") as f:
    xgb_tuned = json.load(f)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []

params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
              early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    Xtr_orig, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr_orig, yva = y[tr_idx], y[va_idx]

    # augment TRAIN portion only with pseudo-labeled rows -- validation stays pure
    Xtr = pd.concat([Xtr_orig, pseudo_X], axis=0, ignore_index=True)
    ytr = np.concatenate([ytr_orig, pseudo_y])

    model = XGBClassifier(**params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred = model.predict_proba(Xva)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(yva, pred)
    fold_aucs.append(auc)
    print(f"fold {fold}: AUC={auc:.5f}  train_size={len(Xtr)}  best_iter={model.best_iteration}")

runtime = time.time() - t0
oof_auc = roc_auc_score(y, oof)
fold_std = float(np.std(fold_aucs))

print(f"\nOOF AUC (on original train rows only): {oof_auc:.5f}")
print(f"fold mean: {np.mean(fold_aucs):.5f}  fold std: {fold_std:.6f}")
print(f"runtime: {runtime:.1f}s")
print(f"\nvs exp022 tuned XGBoost, no pseudo-labels (0.96473): delta = {oof_auc - 0.96473:+.5f}")

np.save("artifacts/oof_exp028_pseudo_labeled.npy", oof)

log_experiment({
    "exp_id": "exp028",
    "model": "XGBoost (tuned, + pseudo-labeling)",
    "features": "9 numeric only",
    "preprocessing": "none (native NaN handling); train folds augmented with 145510 confident (<0.02 or >0.98) test pseudo-labels, validation folds pure original-only",
    "hyperparams": str(params),
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}, SAME split as exp022 for apples-to-apples",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": "pseudo-labeling test, leakage-safe (pseudo rows only in train folds, never validation)",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
