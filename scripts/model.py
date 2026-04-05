import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class MLPRegressor(nn.Module):
    """
    Two-layer MLP head mapping node embeddings to a scalar regression output.
    """

    def __init__(self, in_dim, mlp_hidden=128, dropout=0.2, use_layernorm=True, predict_var=False):
        """
        Build the MLP head.

        Parameters:
        -----------
        in_dim : int
            Input feature dimension per node.
        mlp_hidden : int
            Hidden width of the MLP.
        dropout : float
            Reserved for dropout (not applied in the current ``Sequential``).
        use_layernorm : bool
            Reserved for optional normalization (unused in current implementation).
        predict_var : bool
            whether to predict the variance of the target variable

        Returns:
        --------
        None
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        self.predict_var = predict_var

    def forward(self, x):
        """Map batch of node embeddings ``x`` through the MLP and return shape ``[..., 1]`` logits."""
        return self.mlp(x)

class GraphSAGENet(nn.Module):
    """
    Stacked GraphSAGE convolutions with LayerNorm, SiLU, dropout, and an ``MLPRegressor`` readout.
    """

    def __init__(self, in_channels, hidden_channels=64, num_layers=2, dropout=0.2,
                 mlp_hidden=128, predict_var=False, use_layernorm=True, aggr='mean'):
        """
        Construct SAGEConv stacks and the regression MLP head.

        Parameters:
        -----------
        in_channels : int
            Input node feature size.
        hidden_channels : int
            Hidden size for each SAGE layer.
        num_layers : int
            Number of message-passing layers.
        dropout : float
            Dropout probability after activations.
        mlp_hidden : int
            Hidden width inside ``MLPRegressor``.
        predict_var : bool
            Passed through to ``MLPRegressor``.
        use_layernorm : bool
            If True, apply ``LayerNorm`` after each conv; else ``Identity``.
        aggr : str
            Aggregation scheme for SAGE layers after the first (e.g. ``'mean'``).

        Returns:
        --------
        None
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.aggr = aggr

        # first conv
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels) if use_layernorm else nn.Identity())

        # additional convs
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=self.aggr))
            self.norms.append(nn.LayerNorm(hidden_channels) if use_layernorm else nn.Identity())

        self.dropout = dropout
        self.mlp = MLPRegressor(hidden_channels, mlp_hidden=mlp_hidden,
                               dropout=dropout, use_layernorm=use_layernorm,
                               predict_var=predict_var)
        self.predict_var = predict_var
        self.hidden_channels = hidden_channels

    def forward(self, data):
        """Encode PyG ``data`` (``x``, ``edge_index``) through SAGE layers and return per-node scalar predictions."""
        x, edge_index = data.x, data.edge_index
        for conv, ln in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = ln(x)
            x = F.silu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x: [N, hidden_channels]
        out = self.mlp(x)  # mu or (mu, logvar)
        return out
    
    def get_all_weights(self):
        """
        Collect learnable tensors from conv and MLP submodules.

        Parameters:
        -----------
        None

        Returns:
        --------
        dict
            Maps string names to parameter tensors (CPU data references).
        """
        weights = {}
        
        # Get GraphSAGE layer weights
        for i, conv in enumerate(self.convs):
            if hasattr(conv, 'lin_l') and hasattr(conv, 'lin_r'):
                weights[f'conv_{i}_lin_l_weight'] = conv.lin_l.weight.data
                weights[f'conv_{i}_lin_l_bias'] = conv.lin_l.bias.data
                weights[f'conv_{i}_lin_r_weight'] = conv.lin_r.weight.data
                weights[f'conv_{i}_lin_r_bias'] = conv.lin_r.bias.data
        
        # Get MLP weights
        for name, param in self.mlp.named_parameters():
            weights[f'mlp_{name}'] = param.data
            
        return weights
    
    def get_weight_summary(self):
        """Compute shape/mean/std/min/max stats for every tensor from ``get_all_weights``. Returns a nested summary dict."""
        weights = self.get_all_weights()
        summary = {}
        
        for name, weight in weights.items():
            summary[name] = {
                'shape': weight.shape,
                'mean': weight.mean().item(),
                'std': weight.std().item(),
                'min': weight.min().item(),
                'max': weight.max().item()
            }
        return summary
    
    def predict(self, data, device=None):
        """
        Run inference for all nodes and return CPU predictions.

        Parameters:
        -----------
        data : torch_geometric.data.Data
            Full graph to evaluate.
        device : torch.device or None
            If None, uses the device of the first model parameter.

        Returns:
        --------
        torch.Tensor
            1-D float tensor of predictions aligned with node index order.
        """
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        data = data.to(device)
        
        with torch.no_grad():
            predictions = self(data)
            return predictions.view(-1).cpu()  # Flatten and move to CPU
    
    def predict_train_test(self, data, device=None):
        """
        Split full-graph predictions and targets into train and test subsets.

        Parameters:
        -----------
        data : torch_geometric.data.Data
            Graph with ``train_mask``, ``test_mask``, and ``y``.
        device : torch.device or None
            Inference device; defaults to model parameter device.

        Returns:
        --------
        dict
            Keys ``train_pred``, ``test_pred``, ``train_true``, ``test_true`` mapping to 1-D CPU tensors.
        """
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        data = data.to(device)
        
        with torch.no_grad():
            predictions = self(data)
            predictions = predictions.view(-1).cpu()
            true_values = data.y.view(-1).cpu()
            
            train_mask = data.train_mask.cpu()
            test_mask = data.test_mask.cpu()
            
            return {
                'train_pred': predictions[train_mask],
                'test_pred': predictions[test_mask],
                'train_true': true_values[train_mask],
                'test_true': true_values[test_mask]
            }
