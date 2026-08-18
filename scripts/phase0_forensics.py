"""
Phase 0 — Data forensics for playground-series-s6e8.
Prints a structured report: schema, missingness, cardinality, duplicates,
target distribution, per-feature univariate AUC, and a train-vs-test
adversarial-validation check.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

DATA_DIR = "playground-series-s6e8"
TARGET = "addicted_label"
ID_COL = "id"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

print("=" * 70)
print("SHAPES")
print("=" * 70)
print(f"train: {train.shape}   test: {test.shape}")
print(f"train columns: {list(train.columns)}")
print(f"test columns:  {list(test.columns)}")

feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
# pandas >=3.0 defaults string columns to a "str" extension dtype, not
# object -- dtype == object silently misses them, so use is_string_dtype.
cat_cols = [c for c in feature_cols if pd.api.types.is_string_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in cat_cols]
print(f"\nnumeric features ({len(num_cols)}): {num_cols}")
print(f"categorical features ({len(cat_cols)}): {cat_cols}")

print("\n" + "=" * 70)
print("DTYPES")
print("=" * 70)
print(train.dtypes)

print("\n" + "=" * 70)
print("MISSING VALUES (train)")
print("=" * 70)
miss_train = train.isna().sum().sort_values(ascending=False)
miss_train_pct = (miss_train / len(train) * 100).round(3)
print(pd.DataFrame({"n_missing": miss_train, "pct_missing": miss_train_pct}))

print("\n" + "=" * 70)
print("MISSING VALUES (test)")
print("=" * 70)
miss_test = test.isna().sum().sort_values(ascending=False)
miss_test_pct = (miss_test / len(test) * 100).round(3)
print(pd.DataFrame({"n_missing": miss_test, "pct_missing": miss_test_pct}))

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)
print(f"duplicate id (train): {train[ID_COL].duplicated().sum()}")
print(f"duplicate id (test): {test[ID_COL].duplicated().sum()}")
print(f"fully duplicate rows (train, incl target): {train.drop(columns=[ID_COL]).duplicated().sum()}")
print(f"duplicate feature rows (train, excl target/id): {train[feature_cols].duplicated().sum()}")
print(f"duplicate feature rows (test, excl id): {test[feature_cols].duplicated().sum()}")

# check id ranges / overlap
print(f"\ntrain id range: {train[ID_COL].min()} - {train[ID_COL].max()}")
print(f"test id range: {test[ID_COL].min()} - {test[ID_COL].max()}")
print(f"id overlap between train/test: {len(set(train[ID_COL]) & set(test[ID_COL]))}")

print("\n" + "=" * 70)
print("CARDINALITY / UNIQUE VALUES (categoricals)")
print("=" * 70)
for c in cat_cols:
    vc_train = train[c].value_counts(dropna=False)
    vc_test = test[c].value_counts(dropna=False)
    print(f"\n--- {c} ---")
    print("train:\n", vc_train)
    print("test:\n", vc_test)
    train_only = set(train[c].dropna().unique()) - set(test[c].dropna().unique())
    test_only = set(test[c].dropna().unique()) - set(train[c].dropna().unique())
    print(f"train-only categories: {train_only}")
    print(f"test-only categories: {test_only}")

print("\n" + "=" * 70)
print("NUMERIC SUMMARY (train)")
print("=" * 70)
print(train[num_cols].describe().T)

print("\n" + "=" * 70)
print("NUMERIC SUMMARY (test)")
print("=" * 70)
print(test[num_cols].describe().T)

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)
print(train[TARGET].value_counts(normalize=False))
print(train[TARGET].value_counts(normalize=True))

print("\n" + "=" * 70)
print("UNIVARIATE PREDICTIVE POWER (single-feature LGBM AUC, 3-fold OOF)")
print("=" * 70)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
y = train[TARGET].values

for c in feature_cols:
    X = train[[c]].copy()
    is_cat = c in cat_cols
    if is_cat:
        X[c] = X[c].fillna("missing").astype(str)
    oof = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(X, y):
        model = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            verbosity=-1, random_state=42
        )
        cat_feature = [c] if is_cat else "auto"
        Xtr, Xva = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        if is_cat:
            Xtr[c] = Xtr[c].astype("category")
            Xva[c] = Xva[c].astype("category")
        model.fit(Xtr, y[tr_idx], categorical_feature=cat_feature if is_cat else "auto")
        oof[va_idx] = model.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(y, oof)
    missing_flag = train[c].isna().sum() > 0
    print(f"{c:30s}  AUC={auc:.5f}  missing={missing_flag}")

print("\n" + "=" * 70)
print("MISSINGNESS AS SIGNAL (AUC of is-null indicator alone)")
print("=" * 70)
for c in feature_cols:
    if train[c].isna().sum() == 0:
        continue
    indicator = train[c].isna().astype(int)
    if indicator.nunique() < 2:
        continue
    auc = roc_auc_score(y, indicator)
    print(f"{c:30s}  missing-indicator AUC={auc:.5f}  (0.5=no signal)")

print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION (can a model tell train from test?)")
print("=" * 70)
adv_train = train[feature_cols].copy()
adv_test = test[feature_cols].copy()
adv_train["__is_test__"] = 0
adv_test["__is_test__"] = 1
adv = pd.concat([adv_train, adv_test], axis=0, ignore_index=True)

for c in cat_cols:
    adv[c] = adv[c].fillna("missing").astype("category")

adv_y = adv["__is_test__"].values
adv_X = adv[feature_cols]

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_adv = np.zeros(len(adv))
importances = np.zeros(len(feature_cols))
for tr_idx, va_idx in skf2.split(adv_X, adv_y):
    model = LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        verbosity=-1, random_state=42
    )
    model.fit(adv_X.iloc[tr_idx], adv_y[tr_idx], categorical_feature=cat_cols if cat_cols else "auto")
    oof_adv[va_idx] = model.predict_proba(adv_X.iloc[va_idx])[:, 1]
    importances += model.feature_importances_ / skf2.n_splits

adv_auc = roc_auc_score(adv_y, oof_adv)
print(f"Adversarial validation AUC (train vs test discriminability): {adv_auc:.5f}")
print("(0.5 = train and test are indistinguishable, good; >0.55-0.6 = distribution shift, investigate)")
imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
print(imp_df)

print("\nDONE.")
