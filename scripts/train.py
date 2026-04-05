import os
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import wandb

from process_data import DataProcessor
from model import GraphSAGENet


def train_epoch(model, optimizer, loader, device, predict_var=False):
    """
    Run one NeighborLoader epoch, minimizing MSE on seed (batch root) nodes only.

    Parameters:
    -----------
    model : torch.nn.Module
        Graph model accepting a PyG mini-batch.
    optimizer : torch.optim.Optimizer
        Optimizer for ``model`` parameters.
    loader : torch_geometric.loader.NeighborLoader
        Mini-batch loader yielding subgraph batches with ``batch_size`` seeds.
    device : torch.device
        Device for tensors.
    predict_var : bool
        Reserved for heteroscedastic heads; currently unused.

    Returns:
    --------
    float
        Mean training loss averaged over all seed nodes seen in the epoch.
    """
    model.train()
    total_loss = 0.0
    total_seen = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # model returns mu or (mu, logvar)
        out = model(batch)   # our GraphSAGENet used forward(self, data)

        mu = out.view(-1)
        seed = batch.batch_size
        mu_seed = mu[:seed]
        y_seed = batch.y[:seed].view(-1)
        loss = F.mse_loss(mu_seed, y_seed)

        loss.backward()
        optimizer.step()

        n = y_seed.size(0)
        total_loss += loss.item() * n
        total_seen += n

    return total_loss / total_seen


@torch.no_grad()
def evaluate(model, loader, device, predict_var=False):
    """
    Aggregate predictions and targets across loader batches for seed nodes, then report RMSE.

    Parameters:
    -----------
    model : torch.nn.Module
        Graph model (same contract as ``train_epoch``).
    loader : torch_geometric.loader.NeighborLoader
        Loader over graph batches.
    device : torch.device
        Device for evaluation.
    predict_var : bool
        If True, expects ``(mu, logvar)`` from the model (not used by current net).

    Returns:
    --------
    rmse : float
        Root mean squared error over all seed predictions.
    preds : numpy.ndarray
        Concatenated predicted values.
    trues : numpy.ndarray
        Concatenated ground-truth values.
    """
    model.eval()
    preds = []
    trues = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        if predict_var:
            mu, logvar = out
        else:
            mu = out
        seed = batch.batch_size
        preds.append(mu[:seed].view(-1).cpu())
        trues.append(batch.y[:seed].view(-1).cpu())

    preds = torch.cat(preds)
    trues = torch.cat(trues)
    mse = F.mse_loss(preds, trues).item()
    rmse = np.sqrt(mse)
    return rmse, preds.numpy(), trues.numpy()


def train_epoch_full_graph(model, optimizer, data, device, predict_var=False):
    """
    Perform one full-graph forward/backward step using only ``train_mask`` nodes in the loss.

    Parameters:
    -----------
    model : torch.nn.Module
        Full-graph GNN.
    optimizer : torch.optim.Optimizer
        Optimizer updated once per call.
    data : torch_geometric.data.Data
        Single graph on CPU or GPU with ``train_mask`` and ``y``.
    device : torch.device
        Device for the forward pass.
    predict_var : bool
        Reserved; current model returns a single output tensor.

    Returns:
    --------
    float
        Scalar training loss after the optimizer step.
    """
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    out = model(data)

    if predict_var:
        mu, logvar = out
    else:
        mu = out

    # Use only training nodes
    train_mask = data.train_mask
    mu_train = mu[train_mask].view(-1)
    y_train = data.y[train_mask].view(-1)

    loss = F.mse_loss(mu_train, y_train)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_full_graph(model, data, device, mask_name='val_mask'):
    """
    Evaluate on the full graph for a given split mask.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained GNN.
    data : torch_geometric.data.Data
        Graph with ``y`` and boolean masks.
    device : torch.device
        Evaluation device.
    mask_name : str
        Attribute on ``data`` for the boolean mask (e.g. ``'val_mask'``, ``'test_mask'``).

    Returns:
    --------
    rmse : float
        Root mean squared error on masked nodes.
    mae : float
        Mean absolute error on masked nodes.
    r2 : float
        Coefficient of determination on masked nodes.
    preds : numpy.ndarray
        Model outputs at masked nodes.
    trues : numpy.ndarray
        Targets at masked nodes.
    """
    model.eval()
    data = data.to(device)

    out = model(data)
    mu = out

    mask = getattr(data, mask_name)
    preds = mu[mask].view(-1).cpu()
    trues = data.y[mask].view(-1).cpu()

    mse = F.mse_loss(preds, trues).item()
    mae = F.l1_loss(preds, trues).item()
    rmse = np.sqrt(mse)

    ss_res = ((trues - preds) ** 2).sum().item()
    ss_tot = ((trues - trues.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return rmse, mae, r2, preds.numpy(), trues.numpy()


def parse_args():
    """
    Parse command-line arguments for ``scripts/train.py``.

    Parameters:
    -----------
    None

    Returns:
    --------
    argparse.Namespace
        Parsed arguments, including required ``--config`` path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    return parser.parse_args()


def main():
    """
    Load YAML config, train a GraphSAGE model on FIREbox, save the best weights, and log metrics.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
        Writes ``outputs/best_model.pt`` and prints metrics; optional Weights & Biases logging.
    """
    args = parse_args()

    print("DEBUG: config path =", args.config)
    print("DEBUG: exists =", os.path.exists(args.config))

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ── Reproducibility seeds ─────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb_cfg = cfg.get("wandb", {}) if isinstance(cfg, dict) else {}
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    wandb_run = None

    if wandb_enabled:
        wandb_run = wandb.init(
            project=wandb_cfg.get("project") or os.getenv("WANDB_PROJECT"),
            entity=wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY"),
            name=wandb_cfg.get("name"),
            mode=wandb_cfg.get("mode", "online"),
            config=cfg,
        )
        if wandb_run is not None:
            print(
                "W&B run:",
                f"{wandb_run.entity}/{wandb_run.project}/{wandb_run.name}",
                f"(id={wandb_run.id}, mode={wandb_run.settings.mode})",
            )
            print("W&B URL:", wandb_run.url or "N/A")

    # ── Load + process data ───────────────────────────────────────────────────
    # Use Docker paths if they exist, otherwise use local paths
    if Path("/data").exists():
        DATA_DIR = Path("/data")
        OUT_DIR = Path("/outputs")
    else:
        DATA_DIR = Path("data")
        OUT_DIR = Path("outputs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / cfg["data"]["file"]

    print(f"DEBUG: DATA_DIR = {DATA_DIR}")
    print(f"DEBUG: data_path = {data_path}")
    print(f"DEBUG: data_path exists = {data_path.exists()}")

    data_processor = DataProcessor(file_path=str(data_path))
    # include_Rhalo=True adds halo radius as a node feature
    # k=None (default) → radius-based graph (cKDTree query_pairs with r=1)
    data_processor.create_graph_data(include_Rhalo=True)
    data = data_processor.data

    print(f"Data loaded")
    print(f"Train nodes: {data.train_mask.sum()}")
    print(f"Val   nodes: {data.val_mask.sum()}")
    print(f"Test  nodes: {data.test_mask.sum()}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GraphSAGENet(
        in_channels=data.x.size(1),
        hidden_channels=cfg["model"]["hidden_channels"],
        num_layers=cfg["model"]["num_layers"],
        mlp_hidden=cfg["model"]["mlp_hidden"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"]
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    # Early stopping uses val_mask; test_mask is touched only after training.
    best_val_rmse = float("inf")
    best_model_path = OUT_DIR / "best_model.pt"

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_epoch_full_graph(model, optimizer, data, device)

        # Validation metrics (used for early stopping and logging)
        val_rmse, val_mae, val_r2, _, _ = evaluate_full_graph(
            model, data, device, mask_name='val_mask')

        # Train metrics (for overfitting monitoring)
        train_rmse, train_mae, train_r2, _, _ = evaluate_full_graph(
            model, data, device, mask_name='train_mask')

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)

            if wandb_run is not None:
                artifact = wandb.Artifact(
                    name="best_model",
                    type="model",
                    metadata={
                        "val_rmse": best_val_rmse,
                        "val_mae": val_mae,
                        "val_r2": val_r2,
                        "epoch": epoch,
                    },
                )
                artifact.add_file(str(best_model_path))
                wandb_run.log_artifact(artifact)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "best_val_rmse": best_val_rmse,
            })

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | Train R²: {train_r2:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}"
            )

    print(f"\nBest validation RMSE: {best_val_rmse:.4f}")

    # ── Final test evaluation (one-shot, after training is complete) ──────────
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_rmse, test_mae, test_r2, _, _ = evaluate_full_graph(
        model, data, device, mask_name='test_mask')

    print(f"\n{'='*50}")
    print(f"  Final Test Results (held-out set)")
    print(f"{'='*50}")
    print(f"  Test RMSE : {test_rmse:.4f}")
    print(f"  Test MAE  : {test_mae:.4f}")
    print(f"  Test R²   : {test_r2:.4f}")
    print(f"{'='*50}\n")

    if wandb_run is not None:
        wandb_run.summary["test_rmse"] = test_rmse
        wandb_run.summary["test_mae"]  = test_mae
        wandb_run.summary["test_r2"]   = test_r2
        wandb.finish()


if __name__ == "__main__":
    main()
