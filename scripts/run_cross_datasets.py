"""Standalone script: run 4 cross-dataset experiments (FIREbox/TNG) for 300 epochs
and save the comparison figure to paper/cross_dataset.png.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(SCRIPTS_DIR, '..')
sys.path.insert(0, SCRIPTS_DIR)

from model import GraphSAGENet
from process_data import DataProcessor
from process_tng_data import TNGDataProcessor
from train import train_epoch_full_graph, evaluate_full_graph

FIREBOX_PATH = os.path.join(REPO_ROOT, 'data', 'firebox_data', 'FIREbox_z=0.txt')
TNG_PATH     = os.path.join(REPO_ROOT, 'data', 'tng-data', 'TNG300-1-subhalos_99.parquet')
PAPER_DIR    = os.path.join(REPO_ROOT, 'paper')
os.makedirs(PAPER_DIR, exist_ok=True)

NUM_LAYERS      = 2
HIDDEN_CHANNELS = 64
MLP_HIDDEN      = 128
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {DEVICE}")

# ── Build datasets ────────────────────────────────────────────────────────────
print("Building FIREbox dataset...")
fb_proc = DataProcessor(file_path=FIREBOX_PATH)
fb_proc.create_graph_data(include_Rhalo=False)
firebox_data = fb_proc.data
print(f"  FIREbox: {firebox_data.x.size(0)} nodes, {firebox_data.edge_index.size(1)} edges")

print("Building TNG dataset...")
tng_proc = TNGDataProcessor(file_path=TNG_PATH)
tng_proc.create_graph_data(r=0.8)
tng_data = tng_proc.data
print(f"  TNG:     {tng_data.x.size(0)} nodes, {tng_data.edge_index.size(1)} edges\n")

# ── Experiment runner ─────────────────────────────────────────────────────────
def run_experiment(train_ds_name, test_ds_name):
    train_data = firebox_data if train_ds_name == 'firebox' else tng_data
    test_data  = firebox_data if test_ds_name  == 'firebox' else tng_data

    if train_data.x.size(1) != test_data.x.size(1):
        raise ValueError(f"Feature dim mismatch: {train_data.x.size(1)} vs {test_data.x.size(1)}")

    label = f"{train_ds_name} -> {test_ds_name}"
    print(f"\n=== {label} ===")
    print(f"  Train nodes: {train_data.train_mask.sum().item()}")
    print(f"  Val nodes:   {test_data.val_mask.sum().item()} (for early stopping)")
    print(f"  Test nodes:  {test_data.test_mask.sum().item()}")

    model = GraphSAGENet(in_channels=train_data.x.size(1),
                         hidden_channels=HIDDEN_CHANNELS,
                         num_layers=NUM_LAYERS,
                         mlp_hidden=MLP_HIDDEN).to(DEVICE)
    optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_rmse   = float('inf')
    best_state  = None

    for epoch in range(NUM_EPOCHS):
        loss = train_epoch_full_graph(model, optimizer, train_data, DEVICE)
        # Early stopping: evaluate on test_data val_mask
        val_rmse, val_mae, val_r2, _, _ = evaluate_full_graph(
            model, test_data, DEVICE, mask_name='val_mask')
        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 30 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  Epoch {epoch:03d} | loss={loss:.4f} | val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best val RMSE: {best_rmse:.4f}")

    # Final test-set evaluation
    all_pred = model.predict(test_data, DEVICE)
    all_true = test_data.y.view(-1).cpu()
    mask     = test_data.test_mask.cpu()
    pred, true = all_pred[mask], all_true[mask]
    r2   = r2_score(true.numpy(), pred.numpy())
    rmse = float(torch.sqrt(torch.nn.functional.mse_loss(pred, true)).item())
    mae  = float(torch.nn.functional.l1_loss(pred, true).item())
    print(f"  Test RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    return {"label": label,
            "train": train_ds_name, "test": test_ds_name,
            "rmse": rmse, "mae": mae, "r2": r2,
            "pred": pred, "true": true}

results = [
    run_experiment('firebox', 'firebox'),
    run_experiment('tng',     'tng'),
    run_experiment('firebox', 'tng'),
    run_experiment('tng',     'firebox'),
]

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Cross-Dataset Results (Test Set) ===")
print(f"{'Train':<10} {'Test':<10} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-" * 46)
for r in results:
    print(f"{r['train']:<10} {r['test']:<10} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, r in enumerate(results):
    true, pred = r['true'], r['pred']
    axes[idx].scatter(true, pred, alpha=0.6, s=10, color='orange')
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    axes[idx].plot([lo, hi], [lo, hi], 'r--', alpha=0.8)
    axes[idx].set_xlabel(r'True $\log M_{\rm halo}$')
    axes[idx].set_ylabel(r'Pred $\log M_{\rm halo}$')
    label = r['train'].capitalize() + u' \u2192 ' + r['test'].capitalize()
    axes[idx].set_title(f"{label} (R\u00b2={r['r2']:.4f})")

plt.tight_layout()
out_path = os.path.join(PAPER_DIR, 'cross_dataset.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
