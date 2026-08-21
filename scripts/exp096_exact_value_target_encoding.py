"""
Experiment 096 — Community discussion 733495 reported the single biggest
feature-engineering lever measured in the whole thread: target-encoding
each numeric column by its EXACT value (not treating it as a continuous
magnitude) is worth +0.0027 OOF AUC, bigger than +0.0019 from model
capacity and +0.0007 from the best single engineered feature (our
other_screen). Rationale: "this data was generated and rounded onto a
grid, and encoding the exact value picks the grid up."

Verified this is plausible first: daily_screen_time_hours has only 1389
unique values across 691k rows (~500 rows/value average) -- a real dense
grid, not sparse/unique floats, so exact-value target encoding has enough
samples per bucket to carry real signal rather than just overfit.

Leak-safe methodology: proper K-fold out-of-fold target encoding (each
fold's encoding computed from the OTHER folds only), not a global
encode-then-split which would leak. Tests on top of the current 56-feature
best set via honest 5-fold CV before adopting.
"""
import time
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED


def oof_target_encode(train_col, y, splits, test_col=None, smoothing=10):
    """Leak-safe out-of-fold target encoding by exact value.
    Each fold's encoding is computed from the OTHER folds only."""
    global_mean = y.mean()
    oof_enc = np.full(len(train_col), global_mean)
    for tr_idx, va_idx in splits:
        stats = pd.DataFrame({"val": train_col.iloc[tr_idx].values, "y": y[tr_idx]})
        agg = stats.groupby("val")["y"].agg(["mean", "count"])
        # smoothing toward global mean for low-count buckets
        smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        mapping = smoothed.to_dict()
        oof_enc[va_idx] = train_col.iloc[va_idx].map(mapping).fillna(global_mean).values

    test_enc = None
    if test_col is not None:
        full_stats = pd.DataFrame({"val": train_col.values, "y": y})
        full_agg = full_stats.groupby("val")["y"].agg(["mean", "count"])
        full_smoothed = (full_agg["mean"] * full_agg["count"] + global_mean * smoothing) / (full_agg["count"] + smoothing)
        full_mapping = full_smoothed.to_dict()
        test_enc = test_col.map(full_mapping).fillna(global_mean).values
    return oof_enc, test_enc


def main():
    with open("artifacts/consensus_expanded_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    baseline_feat_cols = cache["feat_cols"]
    train = cache["train"].copy()

    train_raw, test_raw = load_data()
    y = train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))

    new_cols = []
    for c in NUM_COLS:
        oof_enc, _ = oof_target_encode(train_raw[c], y, splits)
        col_name = f"{c}_exactval_te"
        train[col_name] = oof_enc
        new_cols.append(col_name)
    print(f"Added exact-value target-encoded columns: {new_cols}")

    feat_cols_te = baseline_feat_cols + new_cols

    with open("artifacts/exp075_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

    oof_base = np.zeros(len(train))
    oof_te = np.zeros(len(train))
    t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        ytr, yva = y[tr_idx], y[va_idx]

        Xtr_b, Xva_b = train[baseline_feat_cols].iloc[tr_idx], train[baseline_feat_cols].iloc[va_idx]
        m_b = XGBClassifier(**params)
        m_b.fit(Xtr_b, ytr, eval_set=[(Xva_b, yva)], verbose=False)
        oof_base[va_idx] = m_b.predict_proba(Xva_b)[:, 1]

        Xtr_te, Xva_te = train[feat_cols_te].iloc[tr_idx], train[feat_cols_te].iloc[va_idx]
        m_te = XGBClassifier(**params)
        m_te.fit(Xtr_te, ytr, eval_set=[(Xva_te, yva)], verbose=False)
        oof_te[va_idx] = m_te.predict_proba(Xva_te)[:, 1]

        print(f"fold {fold_i}: base={roc_auc_score(yva, oof_base[va_idx]):.5f}  "
              f"+exactval_te={roc_auc_score(yva, oof_te[va_idx]):.5f}  elapsed={time.time()-t0:.0f}s")

    auc_base = roc_auc_score(y, oof_base)
    auc_te = roc_auc_score(y, oof_te)
    print(f"\nBaseline (56 feat) OOF AUC: {auc_base:.5f}")
    print(f"+ exact-value target encoding (9 new cols) OOF AUC: {auc_te:.5f}")
    print(f"Delta: {auc_te - auc_base:+.5f}")

    np.save("artifacts/oof_exp096_exactval_te.npy", oof_te)

    if auc_te > auc_base:
        test = cache["test"].copy()
        for c in NUM_COLS:
            _, test_enc = oof_target_encode(train_raw[c], y, splits, test_col=test_raw[c])
            test[f"{c}_exactval_te"] = test_enc
        with open("artifacts/consensus_exactval_cache.pkl", "wb") as f:
            pickle.dump({"train": train, "test": test, "feat_cols": feat_cols_te}, f)
        print("Saved artifacts/consensus_exactval_cache.pkl (beat baseline, adopting)")

    from common import log_experiment
    log_experiment({
        "exp_id": "exp096",
        "model": "XGBoost (tuned), exact-value target encoding test (leak-safe OOF)",
        "features": f"56-feature baseline + 9 exact-value target-encoded numeric columns",
        "preprocessing": "K-fold out-of-fold target encoding by exact value, smoothing=10 toward global mean",
        "hyperparams": "exp075 tuned params",
        "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV, leak-safe per-fold encoding",
        "cv_mean": f"base={auc_base:.5f} exactval_te={auc_te:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": "sourced from Kaggle discussion 733495, reported +0.0027 OOF AUC in original thread (largest single lever reported) -- cardinality check confirmed dense grid (daily_screen_time_hours: 1389 unique / 691k rows) before testing",
        "conclusion": "TBD",
    })
    print("\nLogged exp096.")


if __name__ == "__main__":
    main()
