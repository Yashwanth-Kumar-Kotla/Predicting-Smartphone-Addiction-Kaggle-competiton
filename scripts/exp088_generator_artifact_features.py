"""
Experiment 088 — Two independently-validated community findings (Kaggle
discussion threads 733541 and 734990/732428), both backed by rigorous
methodology (10-fold ablation, adversarial validation, null-importance
testing on their end):

1. `other_screen` residual: daily_screen_time_hours - (social_media_hours +
   gaming_hours + work_study_hours). Exposes rows where the generator's
   internal arithmetic doesn't add up. Reported +0.00088 OOF AUC, 8.83x
   null-importance baseline in the source thread.
2. Decimal/fractional-part artifacts: the rounding fingerprint left by the
   float generator. Reported 11.68x null-importance baseline for
   daily_screen_time_hours_decimals; independently corroborated by a
   second unrelated thread finding an 8.5-point base-rate swing across the
   first decimal digit of daily_screen_time_hours.

Both are structurally different from anything tried this session (not a
ratio/interaction OpenFE would find, not HPO, not ensembling). Testing on
our own consensus-9 feature set + tuned XGBoost via proper 5-fold CV
before adopting -- same "verify before trusting" discipline as always.
"""
import time
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, DATA_DIR, TARGET, N_FOLDS, SEED

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
        # true text decimal length as written in the CSV, e.g. "3.5" -> 1, "3.567123" -> 6, "3" -> 0
        dec_part = s.str.split(".").str[1]
        df[f"{c}_decimals_len"] = dec_part.str.len().fillna(0).astype(int)
        df[f"{c}_first_digit"] = dec_part.str[0].fillna("-1").astype(int)
    return df


def main():
    with open("artifacts/consensus_data_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    baseline_feat_cols = cache["feat_cols"]

    train_raw, _ = load_data()
    # read as strings to preserve the generator's original decimal precision
    train_str = pd.read_csv(f"{DATA_DIR}/train.csv", dtype=str)

    train_raw_aug = add_arithmetic_artifact(train_raw)
    train_raw_aug = add_decimal_artifact(train_raw_aug, train_str)
    artifact_cols = [c for c in train_raw_aug.columns if c not in train_raw.columns]
    print(f"New artifact feature columns ({len(artifact_cols)}): {artifact_cols}")

    train = cache["train"].copy()
    for c in artifact_cols:
        train[c] = train_raw_aug[c].values

    y = train[TARGET].values
    feat_cols_aug = baseline_feat_cols + artifact_cols

    with open("artifacts/exp075_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))

    oof_base = np.zeros(len(train))
    oof_aug = np.zeros(len(train))
    t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        ytr, yva = y[tr_idx], y[va_idx]

        Xtr_b, Xva_b = train[baseline_feat_cols].iloc[tr_idx], train[baseline_feat_cols].iloc[va_idx]
        m_b = XGBClassifier(**params)
        m_b.fit(Xtr_b, ytr, eval_set=[(Xva_b, yva)], verbose=False)
        oof_base[va_idx] = m_b.predict_proba(Xva_b)[:, 1]

        Xtr_a, Xva_a = train[feat_cols_aug].iloc[tr_idx], train[feat_cols_aug].iloc[va_idx]
        m_a = XGBClassifier(**params)
        m_a.fit(Xtr_a, ytr, eval_set=[(Xva_a, yva)], verbose=False)
        oof_aug[va_idx] = m_a.predict_proba(Xva_a)[:, 1]

        print(f"fold {fold_i}: base={roc_auc_score(yva, oof_base[va_idx]):.5f}  "
              f"aug={roc_auc_score(yva, oof_aug[va_idx]):.5f}  elapsed={time.time()-t0:.0f}s")

    auc_base = roc_auc_score(y, oof_base)
    auc_aug = roc_auc_score(y, oof_aug)
    print(f"\nBaseline (consensus-9, no artifact features) OOF AUC: {auc_base:.5f}")
    print(f"+ generator artifact features ({len(artifact_cols)} new cols) OOF AUC: {auc_aug:.5f}")
    print(f"Delta: {auc_aug - auc_base:+.5f}")

    np.save("artifacts/oof_exp088_artifact_base.npy", oof_base)
    np.save("artifacts/oof_exp088_artifact_aug.npy", oof_aug)

    from common import log_experiment
    log_experiment({
        "exp_id": "exp088",
        "model": "XGBoost (tuned), generator-artifact features test (other_screen residual + decimal/fractional-part features)",
        "features": f"consensus-9 baseline (22) + {len(artifact_cols)} generator-artifact features",
        "preprocessing": "other_screen = daily_screen_time_hours - (social_media+gaming+work_study); decimal-fraction string-length and first-digit per time column",
        "hyperparams": "exp075 tuned params",
        "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV",
        "cv_mean": f"base={auc_base:.5f} aug={auc_aug:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": "sourced from Kaggle discussion threads 733541/734990, independently validated by original authors via 10-fold ablation + adversarial validation + null-importance testing before we tested it ourselves",
        "conclusion": "TBD",
    })
    print("\nLogged exp088.")


if __name__ == "__main__":
    main()
