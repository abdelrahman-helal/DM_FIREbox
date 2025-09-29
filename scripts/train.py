import torch 
import torch.nn.functional as F
import numpy as np

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
    rmse = np.sqrt(mse)
    return rmse, preds.numpy(), trues.numpy()