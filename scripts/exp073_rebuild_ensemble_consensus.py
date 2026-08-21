"""
Experiment 073 — exp071 didn't save its XGBoost consensus-feature OOF to
disk. Recomputing it here (same setup, seed=42) and saving this time, then
rebuilding the ensemble using consensus-9-feature OOF for all 3 models
(XGBoost recomputed here ~0.96721, exp072 LightGBM 0.96653 + CatBoost
0.96659). All full-data CV throughout -- matches deployment methodology.
"""
import time
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from openfe import OpenFE, transform
from scipy.optimize import minimize

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def main():
    train, test = load_data()
    train["social_ratio"] = train["social_media_hours"] / train["daily_screen_time_hours"]
    train["gaming_ratio"] = train["gaming_hours"] / train["daily_screen_time_hours"]
    train["entertainment_ratio"] = train["social_ratio"] + train["gaming_ratio"]
    train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
    current_best_feats = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]

    with open("artifacts/exp070_consensus_features.pkl", "rb") as f:
        consensus_data = pickle.load(f)
    consensus_3_sigs = set(consensus_data["consensus_3_signatures"])

    X = train[NUM_COLS].copy()
    y_df = train[[TARGET]].copy()
    ofe = OpenFE()
    features_seed42 = ofe.fit(data=X, label=y_df, n_jobs=8, seed=42, verbose=False)
    consensus_feats = [f for f in features_seed42 if signature(f) in consensus_3_sigs]

    Xte = test[NUM_COLS].copy()
    X_new, _ = transform(X, Xte, consensus_feats, n_jobs=8)
    new_cols = [c for c in X_new.columns if c not in NUM_COLS]
    for c in new_cols:
        train[c] = X_new[c].values
    feat_cols = current_best_feats + new_cols
    print(f"Total features: {len(feat_cols)}")

    y = train[TARGET].values
    with open("artifacts/exp022_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_xgboost = np.zeros(len(train))
    t0 = time.time()
    for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
        Xtr, Xva = train[feat_cols].iloc[tr_idx], train[feat_cols].iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_xgboost[va_idx] = model.predict_proba(Xva)[:, 1]
    auc_xgb = roc_auc_score(y, oof_xgboost)
    print(f"XGBoost consensus OOF AUC (recomputed): {auc_xgb:.5f}  runtime: {time.time()-t0:.1f}s")
    np.save("artifacts/oof_exp073_xgb_consensus.npy", oof_xgboost)

    oof_lgbm = np.load("artifacts/oof_exp072_lgbm_consensus.npy")
    oof_catboost = np.load("artifacts/oof_exp072_catboost_consensus.npy")

    names = ["lgbm", "catboost", "xgboost"]
    oofs = [oof_lgbm, oof_catboost, oof_xgboost]
    individual_aucs = [roc_auc_score(y, o) for o in oofs]
    print("\nIndividual OOF AUCs (consensus-9 features, all full-data CV):")
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
    print(f"vs exp065 ensemble (top-40 nested, 0.96737): delta = {best_auc - 0.96737:+.5f}")

    np.save("artifacts/oof_exp073_ensemble_consensus.npy", best_blend)
    with open("artifacts/exp073_ensemble_weights.json", "w") as f:
        json.dump(dict(zip(names, best_w.tolist())), f, indent=2)

    from common import log_experiment
    log_experiment({
        "exp_id": "exp073",
        "model": "Ensemble (LGBM+CatBoost+XGBoost, all 3 with consensus-9 features)",
        "features": "current best (13) + 9 consensus OpenFE features, all 3 models",
        "preprocessing": "none",
        "hyperparams": "Nelder-Mead weight optimization on OOF",
        "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV throughout (matches deployment)",
        "cv_mean": f"{best_auc:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": "full 3-model ensemble with consensus-9 features, validation matches deployment methodology",
        "conclusion": "TBD",
    })
    print("\nLogged to experiments/experiment_log.csv")


if __name__ == "__main__":
    main()
