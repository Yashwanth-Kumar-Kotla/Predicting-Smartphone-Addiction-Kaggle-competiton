"""
Experiment 099 — Follow-up to exp096's big win (exact-value target
encoding of raw columns, +0.00072). Tested raw joint-pair encoding first
(e.g. exact (daily_screen_time_hours, social_media_hours) pair) but that
turned out too sparse to be meaningful: only ~3 rows per unique pair vs
~500 for single columns -- would just memorize near-individual rows.

Arithmetic combinations (sums/differences) don't have this problem:
daily_screen_time_hours - social_media_hours has 1288 unique values
across 691k rows (~537 rows/value), essentially as dense as the raw
columns, because combining grid-valued numbers tends to land on a
related grid rather than exploding combinatorially like raw pairing does.

Testing leak-safe OOF exact-value target encoding on 5 arithmetic
combinations on top of the current best (65-feature, exactval-TE'd) set.
"""
import time
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, TARGET, N_FOLDS, SEED


def oof_target_encode(col, y, splits, test_col=None, smoothing=10):
    global_mean = y.mean()
    oof_enc = np.full(len(col), global_mean)
    for tr_idx, va_idx in splits:
        stats = pd.DataFrame({"val": col.iloc[tr_idx].values, "y": y[tr_idx]})
        agg = stats.groupby("val")["y"].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        mapping = smoothed.to_dict()
        oof_enc[va_idx] = col.iloc[va_idx].map(mapping).fillna(global_mean).values

    test_enc = None
    if test_col is not None:
        full_stats = pd.DataFrame({"val": col.values, "y": y})
        full_agg = full_stats.groupby("val")["y"].agg(["mean", "count"])
        full_smoothed = (full_agg["mean"] * full_agg["count"] + global_mean * smoothing) / (full_agg["count"] + smoothing)
        full_mapping = full_smoothed.to_dict()
        test_enc = test_col.map(full_mapping).fillna(global_mean).values
    return oof_enc, test_enc


def build_combos(df):
    return {
        "daily_minus_social": df["daily_screen_time_hours"] - df["social_media_hours"],
        "daily_plus_social": df["daily_screen_time_hours"] + df["social_media_hours"],
        "daily_minus_weekend": df["daily_screen_time_hours"] - df["weekend_screen_time"],
        "daily_minus_ssgw": df["daily_screen_time_hours"] - (df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]),
        "social_minus_gaming": df["social_media_hours"] - df["gaming_hours"],
    }


def main():
    with open("artifacts/consensus_exactval_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    baseline_feat_cols = cache["feat_cols"]
    train = cache["train"].copy()

    train_raw, test_raw = load_data()
    y = train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))

    combos_train = build_combos(train_raw)
    new_cols = []
    for name, col in combos_train.items():
        col_rounded = col.round(6)
        oof_enc, _ = oof_target_encode(col_rounded, y, splits)
        col_name = f"{name}_exactval_te"
        train[col_name] = oof_enc
        new_cols.append(col_name)
    print(f"Added combo exact-value target-encoded columns: {new_cols}")

    feat_cols_combo = baseline_feat_cols + new_cols

    with open("artifacts/exp075_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

    oof_base = np.zeros(len(train))
    oof_combo = np.zeros(len(train))
    t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        ytr, yva = y[tr_idx], y[va_idx]

        Xtr_b, Xva_b = train[baseline_feat_cols].iloc[tr_idx], train[baseline_feat_cols].iloc[va_idx]
        m_b = XGBClassifier(**params)
        m_b.fit(Xtr_b, ytr, eval_set=[(Xva_b, yva)], verbose=False)
        oof_base[va_idx] = m_b.predict_proba(Xva_b)[:, 1]

        Xtr_c, Xva_c = train[feat_cols_combo].iloc[tr_idx], train[feat_cols_combo].iloc[va_idx]
        m_c = XGBClassifier(**params)
        m_c.fit(Xtr_c, ytr, eval_set=[(Xva_c, yva)], verbose=False)
        oof_combo[va_idx] = m_c.predict_proba(Xva_c)[:, 1]

        print(f"fold {fold_i}: base={roc_auc_score(yva, oof_base[va_idx]):.5f}  "
              f"+combo_te={roc_auc_score(yva, oof_combo[va_idx]):.5f}  elapsed={time.time()-t0:.0f}s")

    auc_base = roc_auc_score(y, oof_base)
    auc_combo = roc_auc_score(y, oof_combo)
    print(f"\nBaseline (65 feat) OOF AUC: {auc_base:.5f}")
    print(f"+ combo exact-value target encoding (5 new cols) OOF AUC: {auc_combo:.5f}")
    print(f"Delta: {auc_combo - auc_base:+.5f}")

    np.save("artifacts/oof_exp099_combo_te.npy", oof_combo)

    if auc_combo > auc_base:
        test = cache["test"].copy()
        combos_test = build_combos(test_raw)
        for name, col in combos_train.items():
            col_rounded = col.round(6)
            _, test_enc = oof_target_encode(col_rounded, y, splits, test_col=combos_test[name].round(6))
            test[f"{name}_exactval_te"] = test_enc
        with open("artifacts/consensus_combo_te_cache.pkl", "wb") as f:
            pickle.dump({"train": train, "test": test, "feat_cols": feat_cols_combo}, f)
        print("Saved artifacts/consensus_combo_te_cache.pkl (beat baseline, adopting)")

    from common import log_experiment
    log_experiment({
        "exp_id": "exp099",
        "model": "XGBoost (tuned), arithmetic-combo exact-value target encoding test (leak-safe OOF)",
        "features": "65-feature baseline + 5 combo exact-value target-encoded columns",
        "preprocessing": "leak-safe OOF exact-value target encoding on daily-social, daily+social, daily-weekend, daily-ssgw, social-gaming; smoothing=10",
        "hyperparams": "exp075 tuned params",
        "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV, leak-safe per-fold encoding",
        "cv_mean": f"base={auc_base:.5f} combo_te={auc_combo:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": "raw joint-pair encoding rejected as too sparse (~3 rows/pair) before this; arithmetic combos preserve dense grid (500+ rows/value)",
        "conclusion": "TBD",
    })
    print("\nLogged exp099.")


if __name__ == "__main__":
    main()
