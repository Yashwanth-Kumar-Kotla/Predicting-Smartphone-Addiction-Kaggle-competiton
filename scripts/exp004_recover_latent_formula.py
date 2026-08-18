"""
Experiment 004 — Try to recover the generator's latent formula.

exp003 showed 5 numeric features carry real (monotonic) signal:
daily_screen_time_hours, weekend_screen_time, social_media_hours,
work_study_hours, gaming_hours. If the generator is
p = sigmoid(w . x), a plain logistic regression on just these 5 features
should approach the full LightGBM baseline's OOF AUC (0.96384, exp001).

We also standardize features so coefficients are directly comparable in
magnitude (tells us relative importance in the latent score), and compare
degree-1 (linear) vs degree-2 (adding squared terms) to check for
curvature the pure logistic-in-raw-features model might be missing.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from common import load_data, TARGET

REAL_FEATURES = [
    "daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
    "work_study_hours", "gaming_hours",
]

train, _ = load_data()
sub = train[REAL_FEATURES + [TARGET]].dropna()
print(f"complete-case rows for these 5 features: {len(sub)} / {len(train)} ({len(sub)/len(train)*100:.1f}%)")

X = sub[REAL_FEATURES].values
y = sub[TARGET].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Linear logistic regression on standardized features ---
oof_lin = np.zeros(len(sub))
coefs = []
for tr, va in skf.split(X, y):
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, y[tr])
    oof_lin[va] = lr.predict_proba(Xva)[:, 1]
    coefs.append(lr.coef_[0])

auc_lin = roc_auc_score(y, oof_lin)
mean_coefs = np.mean(coefs, axis=0)
print(f"\nLinear logistic regression (5 features, standardized) OOF AUC: {auc_lin:.5f}")
print("Standardized coefficients (relative weight in latent score):")
for f, c in sorted(zip(REAL_FEATURES, mean_coefs), key=lambda x: -abs(x[1])):
    print(f"  {f:30s} {c:+.4f}")

# --- Add squared terms to check for curvature beyond sigmoid-of-linear ---
oof_quad = np.zeros(len(sub))
for tr, va in skf.split(X, y):
    scaler = StandardScaler().fit(X[tr])
    Xtr_s, Xva_s = scaler.transform(X[tr]), scaler.transform(X[va])
    Xtr_q = np.hstack([Xtr_s, Xtr_s ** 2])
    Xva_q = np.hstack([Xva_s, Xva_s ** 2])
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr_q, y[tr])
    oof_quad[va] = lr.predict_proba(Xva_q)[:, 1]

auc_quad = roc_auc_score(y, oof_quad)
print(f"\nLogistic + squared terms OOF AUC: {auc_quad:.5f}  (delta vs linear: {auc_quad - auc_lin:+.5f})")

print(f"\nFor reference: exp001 full LightGBM (all 12 features) OOF AUC = 0.96384")
print(f"Gap (LGBM full - logistic-5feat-linear): {0.96384 - auc_lin:+.5f}")
