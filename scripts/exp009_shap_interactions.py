"""
Experiment 009 — Use SHAP interaction values from the winning XGBoost
baseline (exp008, OOF AUC 0.96461) to find which feature pairs the model
is already leaning on hardest. exp005/006 showed pairwise interaction
terms recover most of the linear-vs-tree gap among the 5 "real" features
-- this identifies WHICH pairs to hand-craft as explicit features rather
than guessing, so Phase 4 feature engineering is targeted, not blind.

Trains one XGBoost model on a stratified subsample (SHAP interaction
values are O(features^2 * trees), expensive on 691k rows) with the same
hyperparams as exp008, then ranks feature pairs by mean |SHAP interaction|.
"""
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from common import load_data, NUM_COLS, TARGET, SEED

train, _ = load_data()
X_full, y_full = train[NUM_COLS], train[TARGET].values

# Subsample for tractable SHAP interaction computation
X_sub, _, y_sub, _ = train_test_split(
    X_full, y_full, train_size=20000, stratify=y_full, random_state=SEED
)

params = dict(
    n_estimators=1000, learning_rate=0.05, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
    random_state=SEED, n_jobs=-1,
)
model = XGBClassifier(**params)
model.fit(X_sub, y_sub)

explainer = shap.TreeExplainer(model)
shap_interaction = explainer.shap_interaction_values(X_sub.fillna(-999))

# shap_interaction shape: (n_samples, n_features, n_features)
mean_abs_interaction = np.abs(shap_interaction).mean(axis=0)
n = len(NUM_COLS)

pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((NUM_COLS[i], NUM_COLS[j], mean_abs_interaction[i, j]))
pairs.sort(key=lambda x: -x[2])

print("Top feature-pair interactions by mean |SHAP interaction value|:")
for f1, f2, val in pairs[:15]:
    print(f"  {f1:28s} x {f2:28s}  {val:.5f}")

print("\nMain-effect (diagonal) SHAP magnitude per feature:")
diag = [(NUM_COLS[i], mean_abs_interaction[i, i]) for i in range(n)]
diag.sort(key=lambda x: -x[1])
for f, val in diag:
    print(f"  {f:28s} {val:.5f}")
