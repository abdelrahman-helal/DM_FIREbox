"""Standalone script: train three GNN variants (KDTree / WS / BA) for 300 epochs
and save the comparison figure to paper/graph_var.png.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import r2_score

# Resolve paths relative to this script's location
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(SCRIPTS_DIR, '..')
sys.path.insert(0, SCRIPTS_DIR)

from model import GraphSAGENet
from process_multigraphs import DataProcessor
from train import train_epoch_full_graph, evaluate_full_graph

DATA_PATH = os.path.join(REPO_ROOT, 'data', 'firebox_data', 'FIREbox_z=0.txt')
PAPER_DIR = os.path.join(REPO_ROOT, 'paper')
os.makedirs(PAPER_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_LAYERS      = 3
HIDDEN_CHANNELS = 64
MLP_HIDDEN      = 128
LEARNING_RATE   = 0.001
NUM_EPOCHS      = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {DEVICE}")
print(f"Data:   {DATA_PATH}\n")

# ── Build datasets ────────────────────────────────────────────────────────────
print("Building KDTree dataset...")
dp_kdtree = DataProcessor(file_path=DATA_PATH)
dp_kdtree.create_graph_data(graph_type='kdtree',
                             stratify_bins=10, standardize=True)
data_kdtree = dp_kdtree.data

print("Building Watts-Strogatz dataset...")
dp_ws = DataProcessor(file_path=DATA_PATH)
dp_ws.create_graph_data(k=10, graph_type='ws',
                         stratify_bins=10, standardize=True, ws_beta=0.1)
data_ws = dp_ws.data

print("Building Barabasi-Albert dataset...")
dp_ba = DataProcessor(file_path=DATA_PATH)
dp_ba.create_graph_data(k=10, graph_type='ba',
                         stratify_bins=10, standardize=True)
data_ba = dp_ba.data

# ── Initialise models ─────────────────────────────────────────────────────────
def make_model(in_channels):
    return GraphSAGENet(in_channels=in_channels,
                        hidden_channels=HIDDEN_CHANNELS,
                        num_layers=NUM_LAYERS,
                        mlp_hidden=MLP_HIDDEN).to(DEVICE)

model_kdtree = make_model(data_kdtree.x.size(1))
model_ws     = make_model(data_ws.x.size(1))
model_ba     = make_model(data_ba.x.size(1))

# ── Training helper ───────────────────────────────────────────────────────────
def train_model(model, data, tag):
    optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_rmse   = float('inf')
    best_state  = None

    for epoch in range(NUM_EPOCHS):
        loss = train_epoch_full_graph(model, optimizer, data, DEVICE)
        val_rmse, val_mae, val_r2, _, _ = evaluate_full_graph(model, data, DEVICE)
        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 30 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  [{tag}] Epoch {epoch:03d} | loss={loss:.4f} | "
                  f"val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f}")

    model.load_state_dict(best_state)
    print(f"  [{tag}] Best val RMSE: {best_rmse:.4f}\n")
    return best_rmse

print("=== Training KDTree ===")
best_rmse_kdtree = train_model(model_kdtree, data_kdtree, 'KDTree')

print("=== Training Watts-Strogatz ===")
best_rmse_ws = train_model(model_ws, data_ws, 'WS')

print("=== Training Barabasi-Albert ===")
best_rmse_ba = train_model(model_ba, data_ba, 'BA')

# ── Test-set predictions ──────────────────────────────────────────────────────
def test_preds(model, data):
    all_pred = model.predict(data, DEVICE)
    all_true = data.y.view(-1).cpu()
    mask     = data.test_mask.cpu()
    return all_true[mask], all_pred[mask]

true_kd, pred_kd = test_preds(model_kdtree, data_kdtree)
true_ws, pred_ws = test_preds(model_ws,     data_ws)
true_ba, pred_ba = test_preds(model_ba,     data_ba)

r2_kd = r2_score(true_kd.numpy(), pred_kd.numpy())
r2_ws = r2_score(true_ws.numpy(), pred_ws.numpy())
r2_ba = r2_score(true_ba.numpy(), pred_ba.numpy())

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

def scatter_panel(ax, true, pred, r2, title):
    ax.scatter(true, pred, alpha=0.6, s=10, color='orange')
    lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.8)
    ax.set_xlabel(r'True $\log M_{\rm halo}$')
    ax.set_title(title)

scatter_panel(axes[0], true_kd, pred_kd, r2_kd, f'KDTree (R\u00b2={r2_kd:.4f})')
axes[0].set_ylabel(r'Pred $\log M_{\rm halo}$')
scatter_panel(axes[1], true_ws, pred_ws, r2_ws, f'Watts\u2013Strogatz (R\u00b2={r2_ws:.4f})')
scatter_panel(axes[2], true_ba, pred_ba, r2_ba, f'Barab\u00e1si\u2013Albert (R\u00b2={r2_ba:.4f})')

plt.tight_layout()
out_path = os.path.join(PAPER_DIR, 'graph_var.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")

print("\n=== Final R2 Scores (Test Set) ===")
print(f"  KDTree:          R2 = {r2_kd:.4f}  |  best_val_RMSE = {best_rmse_kdtree:.4f}")
print(f"  Watts-Strogatz:  R2 = {r2_ws:.4f}  |  best_val_RMSE = {best_rmse_ws:.4f}")
print(f"  Barabasi-Albert: R2 = {r2_ba:.4f}  |  best_val_RMSE = {best_rmse_ba:.4f}")
