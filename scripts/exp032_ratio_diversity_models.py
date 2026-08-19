"""
Experiment 032 — Test the weekend/daily screen-time ratio (exp031,
independently verified) on our diversity models specifically. The
discussion thread's core claim: this relationship is a HUMP (non-monotone)
that trees already capture via splits (their null result, -0.00007,
matches our exp010 principle) but that monotone models (logistic
regression) structurally cannot represent without an explicit non-monotone
feature. Also testing the GPU MLP, since it never saw the ratio directly
and had to implicitly learn division -- a known weak spot for gradient
nets, distinct from the multiplication our exp027 interaction-term test
covered.

New features (on top of the 9 raw numeric): ratio, ratio^2,
inside_envelope [1.044,1.965] binary, distance_from_envelope (0 if
inside), beyond_2_5x binary.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED

LOW, HIGH = 1.044, 1.965

train, _ = load_data()
y = train[TARGET].values


def engineer_ratio(df):
    ratio = df["weekend_screen_time"] / df["daily_screen_time_hours"]
    inside = ((ratio >= LOW) & (ratio <= HIGH)).astype(float)
    dist = np.where(ratio < LOW, LOW - ratio, np.where(ratio > HIGH, ratio - HIGH, 0.0))
    beyond = (ratio > 2.5).astype(float)
    out = df[NUM_COLS].copy()
    out["ratio"] = ratio
    out["ratio_sq"] = ratio ** 2
    out["inside_envelope"] = inside
    out["dist_from_envelope"] = dist
    out["beyond_2_5x"] = beyond
    return out


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ============================================================
# 1. Logistic regression with ratio features
# ============================================================
print("=" * 70)
print("Logistic regression: 9 raw + ratio-derived features")
print("=" * 70)
oof_lr = np.zeros(len(train))
fold_aucs_lr = []
t0 = time.time()
for tr_idx, va_idx in skf.split(train[NUM_COLS], y):
    Xtr_df = engineer_ratio(train.iloc[tr_idx])
    Xva_df = engineer_ratio(train.iloc[va_idx])
    ytr, yva = y[tr_idx], y[va_idx]

    medians = Xtr_df.median()
    Xtr_df = Xtr_df.fillna(medians)
    Xva_df = Xva_df.fillna(medians)

    scaler = StandardScaler().fit(Xtr_df)
    Xtr = scaler.transform(Xtr_df)
    Xva = scaler.transform(Xva_df)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr, ytr)
    pred = lr.predict_proba(Xva)[:, 1]
    oof_lr[va_idx] = pred
    fold_aucs_lr.append(roc_auc_score(yva, pred))

auc_lr = roc_auc_score(y, oof_lr)
print(f"OOF AUC: {auc_lr:.5f}  fold_std: {np.std(fold_aucs_lr):.6f}  runtime: {time.time()-t0:.1f}s")
print(f"vs exp016 logistic without ratio features (0.92451): delta = {auc_lr - 0.92451:+.5f}")
np.save("artifacts/oof_exp032_logistic_ratio.npy", oof_lr)

# ============================================================
# 2. GPU MLP with ratio features
# ============================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n{'=' * 70}\nGPU MLP: 9 raw + ratio-derived features (device={device})\n{'=' * 70}")
torch.manual_seed(SEED)


class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(Xtr, ytr, Xva, yva, max_epochs=200, patience=20, batch_size=8192):
    model = MLP(Xtr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    n = Xtr_t.shape[0]
    best_auc, best_state, no_improve = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = 1 / (1 + np.exp(-model(Xva_t).cpu().numpy()))
        val_auc = roc_auc_score(yva, val_pred)
        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc, best_state, no_improve = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = 1 / (1 + np.exp(-model(Xva_t).cpu().numpy()))
    return final_pred, epoch + 1


oof_mlp = np.zeros(len(train))
fold_aucs_mlp = []
t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(train[NUM_COLS], y)):
    Xtr_df = engineer_ratio(train.iloc[tr_idx])
    Xva_df = engineer_ratio(train.iloc[va_idx])
    ytr, yva = y[tr_idx], y[va_idx]
    medians = Xtr_df.median()
    Xtr_df = Xtr_df.fillna(medians)
    Xva_df = Xva_df.fillna(medians)
    scaler = StandardScaler().fit(Xtr_df)
    Xtr = scaler.transform(Xtr_df).astype(np.float32)
    Xva = scaler.transform(Xva_df).astype(np.float32)

    pred, n_epochs = train_one_fold(Xtr, ytr, Xva, yva)
    oof_mlp[va_idx] = pred
    auc = roc_auc_score(yva, pred)
    fold_aucs_mlp.append(auc)
    print(f"fold {fold}: AUC={auc:.5f}  epochs={n_epochs}")

auc_mlp = roc_auc_score(y, oof_mlp)
print(f"\nOOF AUC: {auc_mlp:.5f}  fold_std: {np.std(fold_aucs_mlp):.6f}  runtime: {time.time()-t0:.1f}s")
print(f"vs exp026 GPU MLP without ratio features (0.93816): delta = {auc_mlp - 0.93816:+.5f}")
np.save("artifacts/oof_exp032_mlp_ratio.npy", oof_mlp)

# ============================================================
# Blend tests against the tree ensemble
# ============================================================
tree_oof = np.load("artifacts/oof_exp024_ensemble_v2.npy")
from scipy.optimize import minimize

for name, oof in [("logistic+ratio", oof_lr), ("mlp+ratio", oof_mlp)]:
    corr = np.corrcoef(oof, tree_oof)[0, 1]
    def neg_auc(w, oof=oof):
        w = np.abs(w); w = w / w.sum()
        return -roc_auc_score(y, w[0] * tree_oof + w[1] * oof)
    res = minimize(neg_auc, [0.8, 0.2], method="Nelder-Mead")
    best_w = np.abs(res.x); best_w = best_w / best_w.sum()
    blend_auc = roc_auc_score(y, best_w[0] * tree_oof + best_w[1] * oof)
    print(f"\n{name}: corr_with_trees={corr:.5f}  blend_weights=(tree={best_w[0]:.3f}, {name}={best_w[1]:.3f})  blend_auc={blend_auc:.5f}  gain_vs_0.96494={blend_auc-0.96494:+.5f}")

from common import log_experiment
log_experiment({
    "exp_id": "exp032",
    "model": "LogisticRegression + GPU MLP, both with ratio-envelope features",
    "features": "9 raw + ratio, ratio_sq, inside_envelope, dist_from_envelope, beyond_2_5x",
    "preprocessing": "per-fold median imputation + StandardScaler, both fit on train fold only",
    "hyperparams": "logistic: default C=1.0; MLP: same arch as exp026",
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"logistic={auc_lr:.5f} mlp={auc_mlp:.5f}",
    "cv_std": f"logistic={np.std(fold_aucs_lr):.6f} mlp={np.std(fold_aucs_mlp):.6f}",
    "best_fold": "n/a", "worst_fold": "n/a",
    "runtime_sec": f"{time.time()-t0:.1f}",
    "notes": "testing community-sourced weekend/daily ratio hump feature on diversity models specifically (trees already null-tested by the source thread and exp010 principle)",
    "conclusion": "TBD",
})
