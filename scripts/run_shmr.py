"""
Train FIREbox (with/without Rhalo) for 300 epochs,
then plot the Stellar-to-Halo Mass Relation (SHMR) using predicted Mhalo.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from process_data import DataProcessor
from model import GraphSAGENet
from train import train_epoch_full_graph, evaluate_full_graph

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS = 300
HIDDEN = 64
NUM_LAYERS = 2
MLP_HIDDEN = 128
LR = 0.001
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"
PAPER_DIR.mkdir(exist_ok=True)


def train_model(data, epochs=EPOCHS, label=""):
    """
    Train a full-graph GraphSAGE regressor with Adam, keeping the best validation RMSE checkpoint.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        Graph with features, edges, targets, and train/val/test masks.
    epochs : int
        Number of full-graph training steps.
    label : str
        Prefix printed in progress logs.

    Returns:
    --------
    GraphSAGENet
        Model with weights from the lowest validation RMSE epoch (evaluated on test at end).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = GraphSAGENet(
        in_channels=data.x.size(1),
        hidden_channels=HIDDEN,
        num_layers=NUM_LAYERS,
        mlp_hidden=MLP_HIDDEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_epoch_full_graph(model, optimizer, data, DEVICE)
        val_rmse, val_mae, val_r2, _, _ = evaluate_full_graph(
            model, data, DEVICE, mask_name='val_mask')

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            train_rmse, _, train_r2, _, _ = evaluate_full_graph(
                model, data, DEVICE, mask_name='train_mask')
            print(f"  [{label}] Epoch {epoch:03d} | "
                  f"Train RMSE: {train_rmse:.4f} R²: {train_r2:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} R²: {val_r2:.4f}")

    model.load_state_dict(best_state)
    model.to(DEVICE)

    test_rmse, test_mae, test_r2, _, _ = evaluate_full_graph(
        model, data, DEVICE, mask_name='test_mask')
    print(f"  [{label}] Final test RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  R²: {test_r2:.4f}")

    return model


def get_predictions(model, data):
    """
    Predict ``lg_Mhalo`` for all nodes and return parallel NumPy arrays.

    Parameters:
    -----------
    model : GraphSAGENet
        Trained regressor.
    data : torch_geometric.data.Data
        Graph evaluated on ``DEVICE``.

    Returns:
    --------
    preds : numpy.ndarray
        1-D predicted log halo masses.
    true : numpy.ndarray
        1-D ground-truth ``data.y`` values.
    """
    model.eval()
    data_dev = data.to(DEVICE)
    with torch.no_grad():
        preds = model(data_dev).view(-1).cpu().numpy()
    true = data.y.view(-1).cpu().numpy()
    return preds, true


def compute_shmr_binned(lg_mhalo, lg_mstar, n_bins=15, n_boot=1000):
    """
    Bin by halo mass and estimate median stellar fraction with bootstrap uncertainty.

    Parameters:
    -----------
    lg_mhalo : numpy.ndarray
        Log10 halo masses used as binning variable.
    lg_mstar : numpy.ndarray
        Log10 stellar masses aligned with ``lg_mhalo``.
    n_bins : int
        Number of equal-width bins in halo mass.
    n_boot : int
        Bootstrap resamples per bin for median percentiles.

    Returns:
    --------
    bin_centers : numpy.ndarray
        Halo-mass bin centers.
    medians : numpy.ndarray
        Per-bin medians of ``lg_mstar - lg_mhalo``.
    lo16, hi84 : numpy.ndarray
        Bootstrap 16th/84th percentiles around the median.
    counts : numpy.ndarray
        Integer counts of galaxies per bin.
    """
    lg_frac = lg_mstar - lg_mhalo  # log10(M*/Mhalo)

    bin_edges = np.linspace(lg_mhalo.min(), lg_mhalo.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    medians = np.full(n_bins, np.nan)
    lo16 = np.full(n_bins, np.nan)
    hi84 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (lg_mhalo >= bin_edges[i]) & (lg_mhalo < bin_edges[i + 1])
        vals = lg_frac[mask]
        counts[i] = len(vals)
        if len(vals) < 2:
            if len(vals) == 1:
                medians[i] = vals[0]
            continue
        medians[i] = np.median(vals)
        # bootstrap for 16-84 percentile uncertainty on the median
        boot_medians = np.array([
            np.median(np.random.choice(vals, size=len(vals), replace=True))
            for _ in range(n_boot)
        ])
        lo16[i] = np.percentile(boot_medians, 16)
        hi84[i] = np.percentile(boot_medians, 84)

    return bin_centers, medians, lo16, hi84, counts


def plot_shmr(results, save_path):
    """
    Save a two-panel stellar-to-halo mass relation figure (true vs GNN-predicted halo mass).

    Parameters:
    -----------
    results : list of dict
        Each dict has keys ``label``, ``lg_mhalo_true``, ``lg_mhalo_pred``, and ``lg_mstar``.
    save_path : str or pathlib.Path
        Output image path (PNG).

    Returns:
    --------
    None
        Writes the figure to disk and closes it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    colors = ['#1f77b4', '#d62728']
    markers = ['o', 's']

    # Left panel: SHMR using TRUE Mhalo
    ax = axes[0]
    for i, res in enumerate(results):
        bc, med, lo, hi, counts = compute_shmr_binned(
            res['lg_mhalo_true'], res['lg_mstar'])
        valid = counts >= 4
        faint = counts < 4

        yerr_lo = med[valid] - lo[valid]
        yerr_hi = hi[valid] - med[valid]
        ax.errorbar(bc[valid], med[valid],
                    yerr=[yerr_lo, yerr_hi],
                    fmt=markers[i], color=colors[i],
                    capsize=3, markersize=6, label=res['label'])
        # faint symbols for bins with <4 galaxies
        if faint.any():
            ax.scatter(bc[faint], med[faint],
                       marker=markers[i], color=colors[i],
                       alpha=0.3, s=30)

    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{halo}}\;[M_\odot]$', fontsize=13)
    ax.set_ylabel(r'$\log_{10}\, (M_\star / M_{\mathrm{halo}})$', fontsize=13)
    ax.set_title('True $M_{\\mathrm{halo}}$', fontsize=14)
    ax.legend(fontsize=10)

    # Right panel: SHMR using PREDICTED Mhalo
    ax = axes[1]
    for i, res in enumerate(results):
        bc, med, lo, hi, counts = compute_shmr_binned(
            res['lg_mhalo_pred'], res['lg_mstar'])
        valid = counts >= 4
        faint = counts < 4

        yerr_lo = med[valid] - lo[valid]
        yerr_hi = hi[valid] - med[valid]
        ax.errorbar(bc[valid], med[valid],
                    yerr=[yerr_lo, yerr_hi],
                    fmt=markers[i], color=colors[i],
                    capsize=3, markersize=6, label=res['label'])
        if faint.any():
            ax.scatter(bc[faint], med[faint],
                       marker=markers[i], color=colors[i],
                       alpha=0.3, s=30)

    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{halo}}\;[M_\odot]$', fontsize=13)
    ax.set_title('GNN-Predicted $M_{\\mathrm{halo}}$', fontsize=14)
    ax.legend(fontsize=10)

    fig.suptitle('Stellar-to-Halo Mass Relation (SHMR)', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved SHMR plot to {save_path}")
    plt.close(fig)


def main():
    """
    Train Rhalo and no-Rhalo FIREbox models and write the SHMR comparison figure.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
        Saves ``paper/shmr.png`` relative to the repository root.
    """
    print(f"Using device: {DEVICE}")
    results = []

    # ── 1. FIREbox with Rhalo ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training FIREbox (include_Rhalo=True) — 300 epochs")
    print("="*60)
    fb_rhalo = DataProcessor(file_path=str(DATA_DIR / "firebox_data" / "FIREbox_z=0.txt"))
    fb_rhalo.create_graph_data(include_Rhalo=True)
    # save unscaled stellar mass and halo mass before they're lost
    lg_mstar_fb = fb_rhalo.df_filtered['lg_Mstar_<Rhalo'].values.copy()
    lg_mhalo_true_fb = fb_rhalo.df_filtered['lg_Mhalo'].values.copy()

    model_fb_rhalo = train_model(fb_rhalo.data, label="FIREbox+Rhalo")
    preds_fb_rhalo, _ = get_predictions(model_fb_rhalo, fb_rhalo.data)

    results.append({
        'label': 'FIREbox (with $R_{\\mathrm{halo}}$)',
        'lg_mhalo_true': lg_mhalo_true_fb,
        'lg_mhalo_pred': preds_fb_rhalo,
        'lg_mstar': lg_mstar_fb,
    })

    # ── 2. FIREbox without Rhalo ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("Training FIREbox (include_Rhalo=False) — 300 epochs")
    print("="*60)
    fb_no_rhalo = DataProcessor(file_path=str(DATA_DIR / "firebox_data" / "FIREbox_z=0.txt"))
    fb_no_rhalo.create_graph_data(include_Rhalo=False)
    lg_mstar_fb2 = fb_no_rhalo.df_filtered['lg_Mstar_<Rhalo'].values.copy()
    lg_mhalo_true_fb2 = fb_no_rhalo.df_filtered['lg_Mhalo'].values.copy()

    model_fb_no_rhalo = train_model(fb_no_rhalo.data, label="FIREbox-noRhalo")
    preds_fb_no, _ = get_predictions(model_fb_no_rhalo, fb_no_rhalo.data)

    results.append({
        'label': 'FIREbox (no $R_{\\mathrm{halo}}$)',
        'lg_mhalo_true': lg_mhalo_true_fb2,
        'lg_mhalo_pred': preds_fb_no,
        'lg_mstar': lg_mstar_fb2,
    })

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_shmr(results, PAPER_DIR / "shmr.png")
    print("\nDone!")


if __name__ == "__main__":
    main()
