"""
Experiment 092 — exp091's 3-seed consensus discovery (candidate space:
NUM_COLS + 14 artifact columns) found 25 features with full 3/3 consensus,
14 of which (56%) involve an artifact column -- consistent across all 3
independent seeds, unlike the categorical test's 1-off noise. 5 of the 25
overlap with the original consensus-9 (exp070); the other 20 are new.

Materializes those 20 new features (reusing exp091's saved seed-42 Node
objects, no rediscovery needed) on top of the existing 36-feature cache
(base+ratios+consensus-9+artifacts) and evaluates via honest 5-fold CV
before adopting.
"""
import time
import pickle
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from openfe import transform

from common import load_data, DATA_DIR, NUM_COLS, CAT_COLS, TARGET, N_FOLDS, SEED


def signature(feat):
    return (feat.name, tuple(sorted(feat.get_fnode())))


def add_arithmetic_artifact(df):
    df = df.copy()
    df["other_screen"] = df["daily_screen_time_hours"] - (
        df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]
    )
    df["other_screen_abs"] = df["other_screen"].abs()
    return df


def add_decimal_artifact(df, raw_str_df):
    import pandas as pd
    df = df.copy()
    decimal_cols = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
                     "work_study_hours", "sleep_hours", "weekend_screen_time"]
    for c in decimal_cols:
        s = raw_str_df[c].astype(str)
        dec_part = s.str.split(".").str[1]
        df[f"{c}_decimals_len"] = dec_part.str.len().fillna(0).astype(int)
        df[f"{c}_first_digit"] = dec_part.str[0].fillna("-1").astype(int)
    return df


def main():
    import pandas as pd

    with open("artifacts/consensus_artifact_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    baseline_feat_cols = cache["feat_cols"]  # 36: base+ratios+consensus9+artifacts
    train = cache["train"].copy()

    with open("artifacts/exp091_consensus_with_artifacts.pkl", "rb") as f:
        new_data = pickle.load(f)
    with open("artifacts/exp070_consensus_features.pkl", "rb") as f:
        old_data = pickle.load(f)
    old_sigs = set(old_data["consensus_3_signatures"])
    new_sigs = set(new_data["consensus_3_signatures"])
    new_only_sigs = new_sigs - old_sigs
    print(f"New consensus-25 total: {len(new_sigs)}, overlap with old-9: {len(old_sigs & new_sigs)}, "
          f"new-only to materialize: {len(new_only_sigs)}")

    seed42_nodes = new_data["seed42_features"]
    new_only_feats = [f for f in seed42_nodes if signature(f) in new_only_sigs]
    print(f"Matched {len(new_only_feats)}/{len(new_only_sigs)} new-only features from seed42's top-40")

    # rebuild the exact candidate-space dataframe used for exp091 discovery
    train_raw, test_raw = load_data()
    train_str = pd.read_csv(f"{DATA_DIR}/train.csv", dtype=str)
    test_str = pd.read_csv(f"{DATA_DIR}/test.csv", dtype=str)
    train_disc = add_decimal_artifact(add_arithmetic_artifact(train_raw), train_str)
    test_disc = add_decimal_artifact(add_arithmetic_artifact(test_raw), test_str)
    artifact_cols = new_data["artifact_cols"]
    disc_cols = NUM_COLS + artifact_cols
    Xdisc, Xte_disc = train_disc[disc_cols], test_disc[disc_cols]

    X_new, Xte_new = transform(Xdisc, Xte_disc, new_only_feats, n_jobs=4)
    raw_new_cols = [c for c in X_new.columns if c not in disc_cols]
    rename_map = {c: f"newfe_{c}" for c in raw_new_cols}
    new_cols = list(rename_map.values())
    print(f"Materialized columns: {new_cols}")
    for c_old, c_new in rename_map.items():
        train[c_new] = X_new[c_old].values
        Xte_new[c_new] = Xte_new[c_old].values

    feat_cols_expanded = baseline_feat_cols + new_cols
    print(f"Total feature count: {len(feat_cols_expanded)} (was {len(baseline_feat_cols)})")

    y = train[TARGET].values
    with open("artifacts/exp075_best_params.json") as f:
        xgb_tuned = json.load(f)
    params = dict(xgb_tuned, n_estimators=6000, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, random_state=SEED, n_jobs=-1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))

    oof_base = np.zeros(len(train))
    oof_exp = np.zeros(len(train))
    t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        ytr, yva = y[tr_idx], y[va_idx]

        Xtr_b, Xva_b = train[baseline_feat_cols].iloc[tr_idx], train[baseline_feat_cols].iloc[va_idx]
        m_b = XGBClassifier(**params)
        m_b.fit(Xtr_b, ytr, eval_set=[(Xva_b, yva)], verbose=False)
        oof_base[va_idx] = m_b.predict_proba(Xva_b)[:, 1]

        Xtr_e, Xva_e = train[feat_cols_expanded].iloc[tr_idx], train[feat_cols_expanded].iloc[va_idx]
        m_e = XGBClassifier(**params)
        m_e.fit(Xtr_e, ytr, eval_set=[(Xva_e, yva)], verbose=False)
        oof_exp[va_idx] = m_e.predict_proba(Xva_e)[:, 1]

        print(f"fold {fold_i}: base(36)={roc_auc_score(yva, oof_base[va_idx]):.5f}  "
              f"expanded({len(feat_cols_expanded)})={roc_auc_score(yva, oof_exp[va_idx]):.5f}  elapsed={time.time()-t0:.0f}s")

    auc_base = roc_auc_score(y, oof_base)
    auc_exp = roc_auc_score(y, oof_exp)
    print(f"\nBaseline (36 feat) OOF AUC: {auc_base:.5f}")
    print(f"Expanded ({len(feat_cols_expanded)} feat) OOF AUC: {auc_exp:.5f}")
    print(f"Delta: {auc_exp - auc_base:+.5f}")

    if auc_exp > auc_base:
        test = cache["test"].copy()
        for c in new_cols:
            test[c] = Xte_new[c].values
        with open("artifacts/consensus_expanded_cache.pkl", "wb") as f:
            pickle.dump({"train": train, "test": test, "feat_cols": feat_cols_expanded}, f)
        print("Saved artifacts/consensus_expanded_cache.pkl (beat baseline, adopting)")

    np.save("artifacts/oof_exp092_expanded.npy", oof_exp)

    from common import log_experiment
    log_experiment({
        "exp_id": "exp092",
        "model": "XGBoost (tuned), expanded consensus test (new-25 discovery incl. artifact cols, 20 new-only features)",
        "features": f"36-feature baseline (base+ratios+consensus9+artifacts) + {len(new_cols)} new-only consensus features from exp091",
        "preprocessing": "3-seed consensus discovery with NUM_COLS+artifact_cols candidate space",
        "hyperparams": "exp075 tuned params",
        "cv_strategy": "StratifiedKFold n=5 seed=42, full-data CV",
        "cv_mean": f"base={auc_base:.5f} expanded={auc_exp:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": f"14/25 (56%) of exp091's consensus-3 features involved an artifact column, consistent across all 3 seeds -- much stronger signal than the categorical test's 1/40 noise",
        "conclusion": "TBD",
    })
    print("\nLogged exp092.")


if __name__ == "__main__":
    main()
