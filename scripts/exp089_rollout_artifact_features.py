"""
Experiment 089 — exp088 validated generator-artifact features (+0.00022,
every fold improved) on XGBoost alone. Rolling out to LightGBM (exp079
tuned params) and CatBoost (exp023 params), materializing a new cache so
future scripts don't redo this, and rebuilding the ensemble.
"""
import time
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

from common import load_data, DATA_DIR, TARGET, ID_COL, N_FOLDS, SEED

DECIMAL_COLS = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
                "work_study_hours", "sleep_hours", "weekend_screen_time"]


def add_arithmetic_artifact(df):
    df = df.copy()
    df["other_screen"] = df["daily_screen_time_hours"] - (
        df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]
    )
    df["other_screen_abs"] = df["other_screen"].abs()
    return df


def add_decimal_artifact(df, raw_str_df):
    df = df.copy()
    for c in DECIMAL_COLS:
        s = raw_str_df[c].astype(str)
        dec_part = s.str.split(".").str[1]
        df[f"{c}_decimals_len"] = dec_part.str.len().fillna(0).astype(int)
        df[f"{c}_first_digit"] = dec_part.str[0].fillna("-1").astype(int)
    return df


def main():
    with open("artifacts/consensus_data_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    baseline_feat_cols = cache["feat_cols"]

    train_raw, test_raw = load_data()
    train_str = pd.read_csv(f"{DATA_DIR}/train.csv", dtype=str)
    test_str = pd.read_csv(f"{DATA_DIR}/test.csv", dtype=str)

    train_aug = add_decimal_artifact(add_arithmetic_artifact(train_raw), train_str)
    test_aug = add_decimal_artifact(add_arithmetic_artifact(test_raw), test_str)
    artifact_cols = [c for c in train_aug.columns if c not in train_raw.columns]
    print(f"Artifact columns ({len(artifact_cols)}): {artifact_cols}")

    train = cache["train"].copy()
    test = cache["test"].copy()
    for c in artifact_cols:
        train[c] = train_aug[c].values
        test[c] = test_aug[c].values

    feat_cols = baseline_feat_cols + artifact_cols
    print(f"Total feature count: {len(feat_cols)}")

    with open("artifacts/consensus_artifact_cache.pkl", "wb") as f:
        pickle.dump({"train": train, "test": test, "feat_cols": feat_cols}, f)
    print("Saved artifacts/consensus_artifact_cache.pkl")

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
    print(f"\nLightGBM + artifacts OOF AUC: {auc_lgbm:.5f} (vs exp079 no-artifacts 0.96698)")
    print(f"CatBoost + artifacts OOF AUC: {auc_catboost:.5f} (vs exp072 no-artifacts 0.96659)")

    np.save("artifacts/oof_exp089_lgbm_artifact.npy", oof_lgbm)
    np.save("artifacts/oof_exp089_catboost_artifact.npy", oof_catboost)

    # Rebuild ensemble with all 3 models now using artifact features
    oof_xgb = np.load("artifacts/oof_exp088_artifact_aug.npy")
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
    print(f"vs exp080 ensemble (no artifacts, 0.96747): delta = {best_auc - 0.96747:+.5f}")

    np.save("artifacts/oof_exp089_ensemble_artifact.npy", best_blend)
    with open("artifacts/exp089_ensemble_weights.json", "w") as f:
        json.dump(dict(zip(names, best_w.tolist())), f, indent=2)

    from common import log_experiment
    log_experiment({
        "exp_id": "exp089",
        "model": "Ensemble (LGBM+CatBoost+XGBoost, all tuned, consensus-9 + generator-artifact features)",
        "features": f"consensus-9 (22) + {len(artifact_cols)} generator-artifact features, all 3 models",
        "preprocessing": "other_screen residual + decimal/fractional-part artifact features (exp088)",
        "hyperparams": "Nelder-Mead weight optimization on OOF",
        "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV",
        "cv_mean": f"{best_auc:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": f"lgbm={auc_lgbm:.5f} catboost={auc_catboost:.5f} xgboost=0.96754(exp088)",
        "conclusion": "TBD",
    })
    print("\nLogged exp089.")


if __name__ == "__main__":
    main()
