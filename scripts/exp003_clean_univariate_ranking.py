"""
Experiment 003 — Rebuild a trustworthy univariate ranking for ALL numeric
features using raw-value AUC + 1-feature logistic regression (5-fold OOF),
after exp002b showed the earlier LGBM-based univariate scan (exp0) inflated
notifications_per_day / app_opens_per_day via overfitting. This is the
ranking we should actually trust for prioritizing feature engineering.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import load_data, NUM_COLS, TARGET

train, _ = load_data()
results = []

for feat in NUM_COLS:
    sub = train[[feat, TARGET]].dropna()
    x = sub[feat].values
    y = sub[TARGET].values

    raw_auc = roc_auc_score(y, x)
    raw_auc = max(raw_auc, 1 - raw_auc)  # direction-agnostic

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(sub))
    for tr, va in skf.split(x.reshape(-1, 1), y):
        lr = LogisticRegression()
        lr.fit(x[tr].reshape(-1, 1), y[tr])
        oof[va] = lr.predict_proba(x[va].reshape(-1, 1))[:, 1]
    lr_auc = roc_auc_score(y, oof)

    results.append((feat, raw_auc, lr_auc))

results.sort(key=lambda r: -r[2])
print(f"{'feature':30s} {'raw-value AUC':>15s} {'logistic-1feat AUC':>20s}")
for feat, raw_auc, lr_auc in results:
    print(f"{feat:30s} {raw_auc:15.5f} {lr_auc:20.5f}")
