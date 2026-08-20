"""
Experiment 058 — exp054 (+0.00066 XGBoost) and exp057 (+0.00090 LightGBM)
both confirmed nested OpenFE genuinely helps. Completing cross-family
coverage with CatBoost (tuned params from exp023), same leakage-safe
per-fold discovery methodology.
"""
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from openfe import OpenFE, transform

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED, log_experiment


def main():
    train, _ = load_data()
    y_full = train[TARGET].values

    train["social_ratio"] = train["social_media_hours"] / train["daily_screen_time_hours"]
    train["gaming_ratio"] = train["gaming_hours"] / train["daily_screen_time_hours"]
    train["entertainment_ratio"] = train["social_ratio"] + train["gaming_ratio"]
    train["workstudy_ratio"] = train["work_study_hours"] / train["daily_screen_time_hours"]
    current_best_feats = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]

    with open("artifacts/exp023_best_params.json") as f:
        catboost_tuned = json.load(f)
    cb_params = dict(catboost_tuned, iterations=10000, loss_function="Logloss",
                      eval_metric="AUC", random_seed=SEED, early_stopping_rounds=150,
                      verbose=False, task_type="CPU")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    fold_aucs = []

    t_total = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train[NUM_COLS], y_full)):
        fold_t0 = time.time()
        X_tr_raw = train[NUM_COLS].iloc[tr_idx].reset_index(drop=True)
        y_tr_raw = train[[TARGET]].iloc[tr_idx].reset_index(drop=True)
        X_va_raw = train[NUM_COLS].iloc[va_idx].reset_index(drop=True)

        ofe = OpenFE()
        features = ofe.fit(data=X_tr_raw, label=y_tr_raw, n_jobs=8, seed=SEED, verbose=False)
        top_feats = features[:20]

        X_tr_new, X_va_new = transform(X_tr_raw, X_va_raw, top_feats, n_jobs=8)
        new_cols = [c for c in X_tr_new.columns if c not in NUM_COLS]
        print(f"fold {fold}: OpenFE discovery took {time.time()-fold_t0:.1f}s, {len(new_cols)} new cols")

        Xtr_full = train[current_best_feats].iloc[tr_idx].reset_index(drop=True)
        Xva_full = train[current_best_feats].iloc[va_idx].reset_index(drop=True)
        for c in new_cols:
            Xtr_full[c] = X_tr_new[c].values
            Xva_full[c] = X_va_new[c].values

        ytr, yva = y_full[tr_idx], y_full[va_idx]
        model = CatBoostClassifier(**cb_params)
        model.fit(Xtr_full, ytr, eval_set=(Xva_full, yva), use_best_model=True)
        pred = model.predict_proba(Xva_full)[:, 1]
        oof[va_idx] = pred
        auc = roc_auc_score(yva, pred)
        fold_aucs.append(auc)
        print(f"fold {fold}: AUC={auc:.5f}  total_fold_time={time.time()-fold_t0:.1f}s")

    total_auc = roc_auc_score(y_full, oof)
    print(f"\nNested OpenFE CatBoost OOF AUC: {total_auc:.5f}  fold_std: {np.std(fold_aucs):.6f}  total_runtime: {time.time()-t_total:.1f}s")
    print(f"vs CatBoost no-OpenFE baseline (0.96452): delta = {total_auc - 0.96452:+.5f}")

    np.save("artifacts/oof_exp058_nested_openfe_catboost.npy", oof)

    log_experiment({
        "exp_id": "exp058",
        "model": "CatBoost (tuned) + nested OpenFE (fit per-fold, leakage-safe)",
        "features": "current best (13) + top 20 OpenFE features discovered SEPARATELY per fold",
        "preprocessing": "none (native NaN handling)",
        "hyperparams": str(cb_params),
        "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}, OpenFE.fit() re-run inside each fold on train-only data",
        "cv_mean": f"{total_auc:.5f}", "cv_std": f"{np.std(fold_aucs):.6f}",
        "best_fold": f"{max(fold_aucs):.5f}", "worst_fold": f"{min(fold_aucs):.5f}",
        "runtime_sec": f"{time.time()-t_total:.1f}",
        "notes": "completes cross-family nested-OpenFE coverage (XGBoost exp054, LightGBM exp057, now CatBoost)",
        "conclusion": "TBD",
    })
    print("\nLogged to experiments/experiment_log.csv")


if __name__ == "__main__":
    main()
