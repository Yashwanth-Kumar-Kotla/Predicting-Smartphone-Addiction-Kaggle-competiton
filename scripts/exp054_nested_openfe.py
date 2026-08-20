"""
Experiment 054 — Methodologically correct nested OpenFE: exp051's
+0.00106 gain did not survive distillation (exp053, -0.00008), strong
evidence it was inflated by OpenFE selecting its top candidates using the
FULL training label column before our CV loop (selection-bias leakage,
not value leakage). This reruns OpenFE.fit() separately INSIDE each of
the 5 CV folds, using ONLY that fold's training portion -- the validation
fold's labels are never seen during feature discovery or selection. This
is the honest way to test whether automated discovery finds anything
real, at ~5x the compute cost of the single full-data run.
"""
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
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

    with open("artifacts/exp022_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    fold_aucs = []
    all_fold_feature_names = []

    t_total = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train[NUM_COLS], y_full)):
        fold_t0 = time.time()
        X_tr_raw = train[NUM_COLS].iloc[tr_idx].reset_index(drop=True)
        y_tr_raw = train[[TARGET]].iloc[tr_idx].reset_index(drop=True)
        X_va_raw = train[NUM_COLS].iloc[va_idx].reset_index(drop=True)

        # discover features using ONLY this fold's training data -- validation
        # fold's labels are never touched here
        ofe = OpenFE()
        features = ofe.fit(data=X_tr_raw, label=y_tr_raw, n_jobs=8, seed=SEED, verbose=False)
        top_feats = features[:20]

        X_tr_new, X_va_new = transform(X_tr_raw, X_va_raw, top_feats, n_jobs=8)
        new_cols = [c for c in X_tr_new.columns if c not in NUM_COLS]
        all_fold_feature_names.append(new_cols)
        print(f"fold {fold}: OpenFE discovery took {time.time()-fold_t0:.1f}s, {len(new_cols)} new cols")

        Xtr_full = train[current_best_feats].iloc[tr_idx].reset_index(drop=True)
        Xva_full = train[current_best_feats].iloc[va_idx].reset_index(drop=True)
        for c in new_cols:
            Xtr_full[c] = X_tr_new[c].values
            Xva_full[c] = X_va_new[c].values

        ytr, yva = y_full[tr_idx], y_full[va_idx]
        model = XGBClassifier(**params)
        model.fit(Xtr_full, ytr, eval_set=[(Xva_full, yva)], verbose=False)
        pred = model.predict_proba(Xva_full)[:, 1]
        oof[va_idx] = pred
        auc = roc_auc_score(yva, pred)
        fold_aucs.append(auc)
        print(f"fold {fold}: AUC={auc:.5f}  total_fold_time={time.time()-fold_t0:.1f}s")

    total_auc = roc_auc_score(y_full, oof)
    print(f"\nNested OpenFE OOF AUC: {total_auc:.5f}  fold_std: {np.std(fold_aucs):.6f}  total_runtime: {time.time()-t_total:.1f}s")
    print(f"vs current best (0.96538): delta = {total_auc - 0.96538:+.5f}")
    print(f"vs exp051 leaky full-data version (0.96644): delta = {total_auc - 0.96644:+.5f}")

    print("\nFeatures discovered per fold (first 5 per fold):")
    for i, feats in enumerate(all_fold_feature_names):
        print(f"  fold {i}: {feats[:5]}...")

    np.save("artifacts/oof_exp054_nested_openfe.npy", oof)

    log_experiment({
        "exp_id": "exp054",
        "model": "XGBoost (tuned) + nested OpenFE (fit per-fold, leakage-safe)",
        "features": "current best (13) + top 20 OpenFE features discovered SEPARATELY per fold",
        "preprocessing": "none (native NaN handling)",
        "hyperparams": str(params),
        "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}, OpenFE.fit() re-run inside each fold on train-only data",
        "cv_mean": f"{total_auc:.5f}", "cv_std": f"{np.std(fold_aucs):.6f}",
        "best_fold": f"{max(fold_aucs):.5f}", "worst_fold": f"{min(fold_aucs):.5f}",
        "runtime_sec": f"{time.time()-t_total:.1f}",
        "notes": "leakage-safe nested version of exp051, to get an honest answer on whether OpenFE finds real signal",
        "conclusion": "TBD",
    })
    print("\nLogged to experiments/experiment_log.csv")


if __name__ == "__main__":
    main()
