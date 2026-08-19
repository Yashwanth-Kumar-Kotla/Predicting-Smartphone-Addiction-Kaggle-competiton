"""
Experiment 026 — Real PyTorch MLP on Apple M5 GPU (MPS backend), not the
capacity/time-constrained sklearn attempt from exp017 (0.93784 AUC, too
weak to earn ensemble weight despite genuine decorrelation, corr=0.920).

This is deeper (4 hidden layers, BatchNorm, dropout), trained with more
epochs and early stopping on validation AUC, and GPU-accelerated so
iteration is fast. Same 9-feature, per-fold-median-imputed, standardized
input as exp017 for a clean architecture/capacity comparison -- isolates
whether more capacity closes the accuracy gap enough to earn real ensemble
weight this time (exp016/017 both scored zero blend weight due to a 4pt+
accuracy gap vs the tree ensemble).
"""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import load_data, NUM_COLS, TARGET, N_FOLDS, SEED

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(SEED)

train, _ = load_data()
y = train[TARGET].values


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
    best_auc = -1
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_t).cpu().numpy()
        val_pred = 1 / (1 + np.exp(-val_logits))
        val_auc = roc_auc_score(yva, val_pred)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(Xva_t).cpu().numpy()
    final_pred = 1 / (1 + np.exp(-final_logits))
    return final_pred, best_auc, epoch + 1


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
fold_aucs = []

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(train[NUM_COLS], y)):
    Xtr_raw = train[NUM_COLS].iloc[tr_idx].copy()
    Xva_raw = train[NUM_COLS].iloc[va_idx].copy()
    ytr, yva = y[tr_idx], y[va_idx]

    medians = Xtr_raw.median()
    Xtr_raw = Xtr_raw.fillna(medians)
    Xva_raw = Xva_raw.fillna(medians)

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw).astype(np.float32)
    Xva = scaler.transform(Xva_raw).astype(np.float32)

    fold_t0 = time.time()
    pred, val_auc, n_epochs = train_one_fold(Xtr, ytr, Xva, yva)
    oof[va_idx] = pred
    auc = roc_auc_score(yva, pred)
    fold_aucs.append(auc)
    print(f"fold {fold}: AUC={auc:.5f}  epochs={n_epochs}  fold_time={time.time()-fold_t0:.1f}s")

runtime = time.time() - t0
oof_auc = roc_auc_score(y, oof)
fold_std = float(np.std(fold_aucs))

print(f"\nOOF AUC: {oof_auc:.5f}")
print(f"fold mean: {np.mean(fold_aucs):.5f}  fold std: {fold_std:.6f}")
print(f"runtime: {runtime:.1f}s")
print(f"\nvs exp017 sklearn MLP baseline (0.93784): delta = {oof_auc - 0.93784:+.5f}")

np.save("artifacts/oof_exp026_mlp_gpu.npy", oof)

tree_oof = np.load("artifacts/oof_exp024_ensemble_v2.npy")
corr = np.corrcoef(oof, tree_oof)[0, 1]
print(f"\nCorrelation with exp024 tree ensemble OOF: {corr:.5f}")

from scipy.optimize import minimize
def neg_auc(w):
    w = np.abs(w); w = w / w.sum()
    blend = w[0] * tree_oof + w[1] * oof
    return -roc_auc_score(y, blend)
res = minimize(neg_auc, [0.8, 0.2], method="Nelder-Mead")
best_w = np.abs(res.x); best_w = best_w / best_w.sum()
blend_auc = roc_auc_score(y, best_w[0] * tree_oof + best_w[1] * oof)
print(f"Best blend weights: tree_ensemble={best_w[0]:.3f}, mlp_gpu={best_w[1]:.3f}")
print(f"Blended OOF AUC: {blend_auc:.5f}  (vs tree-ensemble-alone 0.96494)")
print(f"Gain from adding GPU MLP to ensemble: {blend_auc - 0.96494:+.5f}")

from common import log_experiment
log_experiment({
    "exp_id": "exp026",
    "model": "PyTorch MLP (MPS GPU, 4 hidden layers + BatchNorm)",
    "features": "9 raw numeric only",
    "preprocessing": "per-fold median imputation + StandardScaler, both fit on train fold only",
    "hyperparams": "hidden=(256,128,64,32), BatchNorm+Dropout(0.2-0.3), AdamW(lr=1e-3,wd=1e-4), ReduceLROnPlateau, batch=8192, early_stopping patience=20",
    "cv_strategy": f"StratifiedKFold n={N_FOLDS} seed={SEED}",
    "cv_mean": f"{oof_auc:.5f}",
    "cv_std": f"{fold_std:.6f}",
    "best_fold": f"{max(fold_aucs):.5f}",
    "worst_fold": f"{min(fold_aucs):.5f}",
    "runtime_sec": f"{runtime:.1f}",
    "notes": f"GPU-accelerated diversity model attempt 3 (real capacity this time); corr with tree ensemble={corr:.5f}; blend AUC={blend_auc:.5f}",
    "conclusion": "TBD",
})
print("\nLogged to experiments/experiment_log.csv")
