"""
Experiment 097 — exp096 validated exact-value target encoding as the
biggest single win of the session (+0.00072, every fold improved
substantially). Rolling out to LightGBM and CatBoost, rebuilding the
ensemble.
"""
import time
import pickle
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

from common import TARGET, N_FOLDS, SEED


def main():
    with open("artifacts/consensus_exactval_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"]
    feat_cols = cache["feat_cols"]
    print(f"Total feature count: {len(feat_cols)}")

    y = train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))

    with open("artifacts/exp079_best_params.json") as f:
        lgbm_tuned = json.load(f)
    with open("artifacts/exp023_best_params.json") as f:
        catboost_tuned = json.load(f)

    oof_lgbm = np.zeros(len(train))
    oof_catboost = np.zeros(len(train))
    t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        lgbm_params = dict(lgbm_tuned, n_estimators=6000, random_state=SEED,
                            verbosity=-1, force_row_wise=True)
        m_lgbm = LGBMClassifier(**lgbm_params)
        m_lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                   callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        oof_lgbm[va_idx] = m_lgbm.predict_proba(Xva)[:, 1]

        cb_params = dict(catboost_tuned, iterations=10000, loss_function="Logloss",
                          eval_metric="AUC", random_seed=SEED, early_stopping_rounds=150,
                          verbose=False, task_type="CPU")
        m_cb = CatBoostClassifier(**cb_params)
        m_cb.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        oof_catboost[va_idx] = m_cb.predict_proba(Xva)[:, 1]

        print(f"fold {fold_i}: lgbm={roc_auc_score(yva, oof_lgbm[va_idx]):.5f}  "
              f"catboost={roc_auc_score(yva, oof_catboost[va_idx]):.5f}  elapsed={time.time()-t0:.0f}s")

    auc_lgbm = roc_auc_score(y, oof_lgbm)
    auc_catboost = roc_auc_score(y, oof_catboost)
    print(f"\nLightGBM + exactval_te OOF AUC: {auc_lgbm:.5f} (vs exp093 pre-exactval 0.96726)")
    print(f"CatBoost + exactval_te OOF AUC: {auc_catboost:.5f} (vs exp093 pre-exactval 0.96686)")

    np.save("artifacts/oof_exp097_lgbm_exactval.npy", oof_lgbm)
    np.save("artifacts/oof_exp097_catboost_exactval.npy", oof_catboost)

    oof_xgb = np.load("artifacts/oof_exp096_exactval_te.npy")
    names = ["lgbm", "catboost", "xgboost"]
    oofs = [oof_lgbm, oof_catboost, oof_xgb]
    for n, o in zip(names, oofs):
        print(f"  {n:10s} {roc_auc_score(y, o):.5f}")

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
    print(f"vs exp093 ensemble (pre-exactval, 0.96785): delta = {best_auc - 0.96785:+.5f}")

    np.save("artifacts/oof_exp097_ensemble_exactval.npy", best_blend)
    with open("artifacts/exp097_ensemble_weights.json", "w") as f:
        json.dump(dict(zip(names, best_w.tolist())), f, indent=2)

    from common import log_experiment
    log_experiment({
        "exp_id": "exp097",
        "model": "Ensemble (LGBM+CatBoost+XGBoost, all tuned, + exact-value target encoding)",
        "features": f"{len(feat_cols)} features (56-feature expanded set + 9 exact-value target-encoded), all 3 models",
        "preprocessing": "leak-safe OOF exact-value target encoding, smoothing=10",
        "hyperparams": "Nelder-Mead weight optimization on OOF",
        "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV",
        "cv_mean": f"{best_auc:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": f"lgbm={auc_lgbm:.5f} catboost={auc_catboost:.5f} xgboost=0.96843(exp096)",
        "conclusion": "TBD",
    })
    print("\nLogged exp097.")


if __name__ == "__main__":
    main()
