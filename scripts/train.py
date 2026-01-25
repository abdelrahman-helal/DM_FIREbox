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
    """Train on the full graph without using NeighborLoader"""
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
def evaluate_full_graph(model, data, device, predict_var=False):
    """Evaluate on the full graph without using NeighborLoader"""
    model.eval()
    data = data.to(device)
    
    out = model(data)
    if predict_var:
        mu, logvar = out
    else:
        mu = out
    
    # Use only test nodes
    test_mask = data.test_mask
    preds = mu[test_mask].view(-1).cpu()
    trues = data.y[test_mask].view(-1).cpu()
    
    mse = F.mse_loss(preds, trues).item()
    mae = F.l1_loss(preds, trues).item()
    rmse = np.sqrt(mse)
    return rmse, mae, preds.numpy(), trues.numpy()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("DEBUG: config path =", args.config)
    print("DEBUG: exists =", os.path.exists(args.config))

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

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

    # Load + process data
    # Use Docker paths if they exist, otherwise use local paths
    if Path("/data").exists():
        DATA_DIR = Path("/data")
        OUT_DIR = Path("/outputs")
    else:
        # Running locally
        DATA_DIR = Path("data")
        OUT_DIR = Path("outputs")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / cfg["data"]["file"]
    
    print(f"DEBUG: DATA_DIR = {DATA_DIR}")
    print(f"DEBUG: data_path = {data_path}")
    print(f"DEBUG: data_path exists = {data_path.exists()}")

    data_processor = DataProcessor(file_path=str(data_path))
    data_processor.create_graph_data()
    data = data_processor.data

    print(f"Data loaded")
    print(f"Train nodes: {data.train_mask.sum()}")
    print(f"Test nodes: {data.test_mask.sum()}")

    # Model
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

    # Training loop
    best_val_rmse = float("inf")
    best_model_path = OUT_DIR / "best_model.pt"

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_epoch_full_graph(
            model, optimizer, data, device
        )

        val_rmse, val_mae, _, _ = evaluate_full_graph(
            model, data, device
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                model.state_dict(),
                best_model_path
            )

            if wandb_run is not None:
                artifact = wandb.Artifact(
                    name="best_model",
                    type="model",
                    metadata={
                        "val_rmse": best_val_rmse,
                        "val_mae": val_mae,
                        "epoch": epoch,
                    },
                )
                artifact.add_file(str(best_model_path))
                wandb_run.log_artifact(artifact)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "best_val_rmse": best_val_rmse,
                }
            )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )

    print(f"Best validation RMSE: {best_val_rmse:.4f}")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()