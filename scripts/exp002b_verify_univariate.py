"""
Sanity check on exp0's univariate AUCs for notifications_per_day and
app_opens_per_day: the qcut binning showed a flat, non-monotonic target
rate (~0.6-0.8 band) for these two features, which is hard to reconcile
with a reported LGBM univariate AUC of ~0.73-0.74. Verify with a simple,
low-variance estimator (raw-value ranking AUC, ignoring NaNs) to check
whether the earlier depth-4/200-round LGBM number was inflated by
overfitting noise rather than real signal.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import load_data, TARGET

train, _ = load_data()
y_full = train[TARGET].values

for feat in ["notifications_per_day", "app_opens_per_day", "daily_screen_time_hours", "social_media_hours"]:
    sub = train[[feat, TARGET]].dropna()
    x = sub[feat].values
    y = sub[TARGET].values

    # 1. raw-value-as-score AUC (captures monotonic signal only)
    raw_auc = roc_auc_score(y, x)

    # 2. simple 1-feature logistic regression, 5-fold OOF (linear signal only)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(sub))
    for tr, va in skf.split(x.reshape(-1, 1), y):
        lr = LogisticRegression()
        lr.fit(x[tr].reshape(-1, 1), y[tr])
        oof[va] = lr.predict_proba(x[va].reshape(-1, 1))[:, 1]
    lr_auc = roc_auc_score(y, oof)

    print(f"{feat:30s}  raw-value AUC={raw_auc:.5f}   logistic-1feat AUC={lr_auc:.5f}")
