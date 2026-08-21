"""
Experiment 090 — Final submission using exp089's ensemble: consensus-9 +
14 generator-artifact features (other_screen residual + decimal-fraction
artifacts), all 3 models tuned. CV = 0.96771. Multi-seed bagging (3 seeds)
as established.
"""
import time
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from common import TARGET, ID_COL, SEED


def main():
    with open("artifacts/consensus_artifact_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"]
    test = cache["test"]
    feat_cols = cache["feat_cols"]
    print(f"Total feature count: {len(feat_cols)}")

    yfull = train[TARGET].values

    with open("artifacts/exp079_best_params.json") as f:
        lgbm_tuned = json.load(f)
    with open("artifacts/exp075_best_params.json") as f:
        xgb_tuned = json.load(f)
    with open("artifacts/exp023_best_params.json") as f:
        catboost_tuned = json.load(f)
    with open("artifacts/exp089_ensemble_weights.json") as f:
        weights = json.load(f)

    SEEDS = [42, 202, 2026]
    preds = {"lgbm": [], "catboost": [], "xgboost": []}

    t0 = time.time()
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        Xtr, Xval, ytr, yval = train_test_split(
            train[feat_cols], yfull, test_size=0.05, stratify=yfull, random_state=seed
        )

        print("LightGBM (HPO-tuned)...")
        lgbm_params = dict(lgbm_tuned, n_estimators=6000, random_state=seed,
                            verbosity=-1, force_row_wise=True)
        m = LGBMClassifier(**lgbm_params)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="auc",
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        preds["lgbm"].append(m.predict_proba(test[feat_cols])[:, 1])
        print(f"  best_iter={m.best_iteration_}")

        print("CatBoost...")
        cb_params = dict(catboost_tuned, iterations=10000, loss_function="Logloss",
                          eval_metric="AUC", random_seed=seed, early_stopping_rounds=150,
                          verbose=False, task_type="CPU")
        m = CatBoostClassifier(**cb_params)
        m.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
        preds["catboost"].append(m.predict_proba(test[feat_cols])[:, 1])
        print(f"  best_iter={m.get_best_iteration()}")

        print("XGBoost (HPO-tuned)...")
        xgb_params = dict(xgb_tuned, n_estimators=7000, tree_method="hist",
                           eval_metric="auc", early_stopping_rounds=100,
                           random_state=seed, n_jobs=-1)
        m = XGBClassifier(**xgb_params)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        preds["xgboost"].append(m.predict_proba(test[feat_cols])[:, 1])
        print(f"  best_iter={m.best_iteration}")

        print(f"seed {seed} done, elapsed {time.time()-t0:.0f}s")

    pred_lgbm = np.mean(preds["lgbm"], axis=0)
    pred_catboost = np.mean(preds["catboost"], axis=0)
    pred_xgb = np.mean(preds["xgboost"], axis=0)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

    np.save("artifacts/pred_test_exp090_lgbm.npy", pred_lgbm)
    np.save("artifacts/pred_test_exp090_catboost.npy", pred_catboost)
    np.save("artifacts/pred_test_exp090_xgboost.npy", pred_xgb)

    print(f"Blend weights (from exp089): {weights}")

    final_pred = (
        weights["lgbm"] * pred_lgbm
        + weights["catboost"] * pred_catboost
        + weights["xgboost"] * pred_xgb
    )
    final_pred = np.clip(final_pred, 0.0, 1.0)

    sample_sub = pd.read_csv("playground-series-s6e8/sample_submission.csv")
    assert len(final_pred) == len(test)
    assert (test[ID_COL].values == sample_sub[ID_COL].values).all()
    assert np.isfinite(final_pred).all()
    assert (final_pred >= 0).all() and (final_pred <= 1).all()

    submission = pd.DataFrame({"id": test[ID_COL].values, "addicted_label": final_pred})
    submission.to_csv("submission_v12.csv", index=False)

    old_sub = pd.read_csv("submission_v11.csv")
    diff = (submission["addicted_label"] - old_sub["addicted_label"]).abs()
    print(f"\nAll Phase 15 sanity checks passed.")
    print(f"submission_v12.csv written: {len(submission)} rows")
    print(f"mean |change| vs submission_v11.csv (prev): {diff.mean():.5f}")
    print(f"correlation with submission_v11.csv: {submission['addicted_label'].corr(old_sub['addicted_label']):.6f}")
    print(f"\nprediction stats: min={final_pred.min():.5f} max={final_pred.max():.5f} mean={final_pred.mean():.5f}")
    print(f"(train target mean for reference: {yfull.mean():.5f})")


if __name__ == "__main__":
    main()
