"""
Experiment 085 — Per-model HPO (exp075/079/081/083) and ensemble stacking
(exp084) are both exhausted with only marginal/no further gains. Testing a
fundamentally different lever: pseudo-labeling. This is a synthetic
Playground Series dataset where test almost certainly comes from the same
generator/distribution as train, making confident test pseudo-labels a
plausible source of genuine additional signal rather than noise.

Methodology (leakage-safe): select high-confidence test rows from the
session-best ensemble's test predictions (exp082, prob < 0.02 or > 0.98).
For each of the 5 original CV folds, train on (real train fold ∪ ALL
pseudo-labeled test rows), but validate ONLY on the held-out real train
slice -- pseudo-labels never touch the validation set, so the reported AUC
is an honest read against real ground truth.

Uses tuned XGBoost (exp075 params) as the test model since it's the
strongest single model and fastest of the three tuned models.
"""
import time
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import TARGET, N_FOLDS, SEED


def main():
    with open("artifacts/consensus_data_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"]
    test = cache["test"]
    feat_cols = cache["feat_cols"]

    y = train[TARGET].values
    Xd = train[feat_cols]

    with open("artifacts/exp080_ensemble_weights.json") as f:
        weights = json.load(f)
    pred_lgbm = np.load("artifacts/pred_test_exp082_lgbm.npy")
    pred_catboost = np.load("artifacts/pred_test_exp082_catboost.npy")
    pred_xgb = np.load("artifacts/pred_test_exp082_xgboost.npy")
    test_ensemble_pred = (
        weights["lgbm"] * pred_lgbm + weights["catboost"] * pred_catboost + weights["xgboost"] * pred_xgb
    )

    confident_mask = (test_ensemble_pred < 0.02) | (test_ensemble_pred > 0.98)
    n_confident = confident_mask.sum()
    print(f"Confident test rows (prob<0.02 or >0.98): {n_confident} / {len(test)} ({100*n_confident/len(test):.1f}%)")

    Xpseudo = test.loc[confident_mask, feat_cols]
    ypseudo = (test_ensemble_pred[confident_mask] > 0.5).astype(int)
    print(f"Pseudo-label class balance: positive={ypseudo.mean():.4f} (train mean={y.mean():.4f})")

    with open("artifacts/exp075_best_params.json") as f:
        xgb_tuned = json.load(f)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(Xd, y))

    oof_baseline = np.zeros(len(train))
    oof_pseudo = np.zeros(len(train))
    t0 = time.time()

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        Xtr, Xva = Xd.iloc[tr_idx], Xd.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                      early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

        # Baseline: real train fold only
        m_base = XGBClassifier(**params)
        m_base.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_baseline[va_idx] = m_base.predict_proba(Xva)[:, 1]

        # Pseudo-labeled: real train fold + ALL confident pseudo-labels, validated on real fold only
        Xtr_aug = pd.concat([Xtr, Xpseudo], axis=0, ignore_index=True)
        ytr_aug = np.concatenate([ytr, ypseudo])
        m_pseudo = XGBClassifier(**params)
        m_pseudo.fit(Xtr_aug, ytr_aug, eval_set=[(Xva, yva)], verbose=False)
        oof_pseudo[va_idx] = m_pseudo.predict_proba(Xva)[:, 1]

        print(f"fold {fold_i}: baseline={roc_auc_score(yva, oof_baseline[va_idx]):.5f}  "
              f"pseudo={roc_auc_score(yva, oof_pseudo[va_idx]):.5f}  elapsed={time.time()-t0:.0f}s")

    auc_baseline = roc_auc_score(y, oof_baseline)
    auc_pseudo = roc_auc_score(y, oof_pseudo)
    print(f"\nBaseline (real train only) OOF AUC: {auc_baseline:.5f}")
    print(f"Pseudo-labeled (train + {n_confident} confident test rows) OOF AUC: {auc_pseudo:.5f}")
    print(f"Delta: {auc_pseudo - auc_baseline:+.5f}")

    np.save("artifacts/oof_exp085_pseudo_baseline.npy", oof_baseline)
    np.save("artifacts/oof_exp085_pseudo_augmented.npy", oof_pseudo)

    from common import log_experiment
    log_experiment({
        "exp_id": "exp085",
        "model": "XGBoost (tuned) with pseudo-labeling vs baseline, leakage-safe fold-wise comparison",
        "features": "current best (13) + 9 consensus OpenFE features",
        "preprocessing": f"pseudo-labels: {n_confident} confident test rows (prob<0.02 or >0.98) added to each fold's training set only, never to validation",
        "hyperparams": "exp075 tuned params, n_estimators=6000 early_stopping=100",
        "cv_strategy": "StratifiedKFold n=5 seed=42, validated on real train only (pseudo-labels excluded from validation)",
        "cv_mean": f"baseline={auc_baseline:.5f} pseudo={auc_pseudo:.5f}", "cv_std": "n/a",
        "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": "testing whether confident test pseudo-labels add genuine signal (synthetic generator, test likely same distribution as train)",
        "conclusion": "TBD",
    })
    print("\nLogged exp085.")


if __name__ == "__main__":
    main()
