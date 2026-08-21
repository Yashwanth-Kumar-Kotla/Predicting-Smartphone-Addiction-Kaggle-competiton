"""
Experiment 086 — Phase 0 forensics showed gender/stress_level/academic_work_impact
carry ~zero standalone signal (AUC 0.50-0.51), which is why every OpenFE
discovery run this session (exp050 onward) restricted its candidate space to
NUM_COLS only. But that means GroupBy-style interaction features (e.g. mean
daily_screen_time_hours BY gender) were never actually searched -- a
univariate-noise categorical can still carry real signal in combination with
a numeric column, and OpenFE's GroupByThenMean/GroupByThenRank/etc. operators
specifically target exactly this pattern.

This is a single-seed EXPLORATORY screen (not the full 3-seed consensus
methodology) to cheaply check whether this unexplored corner has anything
worth pursuing before investing in a proper leak-safe consensus run. Uses
nested-style evaluation (candidates ranked on train-fold only, evaluated on
held-out fold) on a single 80/20 split to get a rough, less-biased signal.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from openfe import OpenFE, transform

from common import load_data, NUM_COLS, CAT_COLS, TARGET, SEED


def main():
    train, test = load_data()
    for df in (train, test):
        df["social_ratio"] = df["social_media_hours"] / df["daily_screen_time_hours"]
        df["gaming_ratio"] = df["gaming_hours"] / df["daily_screen_time_hours"]
        df["entertainment_ratio"] = df["social_ratio"] + df["gaming_ratio"]
        df["workstudy_ratio"] = df["work_study_hours"] / df["daily_screen_time_hours"]
    current_best_feats = NUM_COLS + ["entertainment_ratio", "social_ratio", "gaming_ratio", "workstudy_ratio"]

    with open("artifacts/consensus_data_cache.pkl", "rb") as f:
        import pickle
        cache = pickle.load(f)
    baseline_feat_cols = cache["feat_cols"]
    consensus_cols = [c for c in baseline_feat_cols if c not in current_best_feats]
    # Merge the already-materialized consensus columns onto `train` by row position
    # (cache["train"] was built from the same load_data() call, same row order/index).
    for c in consensus_cols:
        train[c] = cache["train"][c].values

    # Nested split: discover candidates on train_disc only, evaluate the resulting
    # features (fit on train_disc, scored on held-out val_disc) to avoid selection bias.
    y = train[TARGET].values
    train_disc, val_disc = train_test_split(train, test_size=0.2, stratify=y, random_state=SEED)

    disc_cols = NUM_COLS + CAT_COLS
    Xdisc = train_disc[disc_cols].copy()
    ydisc = train_disc[[TARGET]].copy()

    t0 = time.time()
    ofe = OpenFE()
    all_feats = ofe.fit(data=Xdisc, label=ydisc, n_jobs=8, seed=SEED, verbose=False,
                          categorical_features=CAT_COLS)
    print(f"OpenFE discovery (with categoricals) took {time.time()-t0:.1f}s, found {len(all_feats)} candidates")

    top_feats = all_feats[:40]
    cat_involved = [f for f in top_feats if any(c in f.get_fnode() for c in CAT_COLS)]
    print(f"Of top 40: {len(cat_involved)} involve a categorical column")
    for f in cat_involved[:15]:
        print(f"  {f.name}({f.get_fnode()})")

    Xval = val_disc[disc_cols].copy()
    X_new, Xval_new = transform(Xdisc, Xval, top_feats, n_jobs=8)
    raw_new_cols = [c for c in X_new.columns if c not in disc_cols]
    rename_map = {c: f"catfe_{c}" for c in raw_new_cols}
    new_cols = list(rename_map.values())

    train_disc_aug = train_disc.copy()
    val_disc_aug = val_disc.copy()
    for c_old, c_new in rename_map.items():
        train_disc_aug[c_new] = X_new[c_old].values
        val_disc_aug[c_new] = Xval_new[c_old].values

    feat_cols_with_cat = baseline_feat_cols + new_cols

    # Baseline (no new categorical-interaction features) vs augmented, same nested split
    m_base = XGBClassifier(n_estimators=2000, max_depth=6, learning_rate=0.06,
                            tree_method="hist", eval_metric="auc",
                            early_stopping_rounds=75, random_state=SEED, n_jobs=-1)
    m_base.fit(train_disc_aug[baseline_feat_cols], train_disc_aug[TARGET],
               eval_set=[(val_disc_aug[baseline_feat_cols], val_disc_aug[TARGET])], verbose=False)
    auc_base = roc_auc_score(val_disc_aug[TARGET], m_base.predict_proba(val_disc_aug[baseline_feat_cols])[:, 1])

    m_aug = XGBClassifier(n_estimators=2000, max_depth=6, learning_rate=0.06,
                           tree_method="hist", eval_metric="auc",
                           early_stopping_rounds=75, random_state=SEED, n_jobs=-1)
    m_aug.fit(train_disc_aug[feat_cols_with_cat], train_disc_aug[TARGET],
              eval_set=[(val_disc_aug[feat_cols_with_cat], val_disc_aug[TARGET])], verbose=False)
    auc_aug = roc_auc_score(val_disc_aug[TARGET], m_aug.predict_proba(val_disc_aug[feat_cols_with_cat])[:, 1])

    print(f"\nNested holdout AUC, baseline (consensus-9 only): {auc_base:.5f}")
    print(f"Nested holdout AUC, + top-40 candidates incl. categoricals: {auc_aug:.5f}")
    print(f"Delta: {auc_aug - auc_base:+.5f}")

    from common import log_experiment
    log_experiment({
        "exp_id": "exp086",
        "model": "XGBoost, exploratory: OpenFE discovery WITH categorical columns included (single-seed nested holdout)",
        "features": f"consensus-9 baseline + top-40 OpenFE candidates from NUM_COLS+CAT_COLS search ({len(new_cols)} materialized)",
        "preprocessing": "categorical_features=CAT_COLS passed to OpenFE.fit()",
        "hyperparams": "n_estimators=2000 max_depth=6 lr=0.06 early_stopping=75 (untuned, exploratory)",
        "cv_strategy": "single 80/20 nested holdout (discover on 80%, evaluate on held-out 20%) -- NOT the full 5-fold/3-seed consensus methodology, exploratory only",
        "cv_mean": f"base={auc_base:.5f} aug={auc_aug:.5f}", "cv_std": "n/a", "best_fold": "n/a", "worst_fold": "n/a",
        "runtime_sec": f"{time.time()-t0:.1f}",
        "notes": f"{len(cat_involved)}/40 top candidates involved a categorical column despite categoricals having ~0.50 standalone AUC (phase0 forensics) -- testing if GroupBy-style interactions add signal beyond univariate noise",
        "conclusion": "TBD",
    })
    print("\nLogged exp086.")


if __name__ == "__main__":
    main()
