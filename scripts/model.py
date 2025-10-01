import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, mlp_hidden=128, dropout=0.2, use_layernorm=True, predict_var=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        self.predict_var = predict_var

    def forward(self, x):
        return self.mlp(x)

class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, dropout=0.2,
                 mlp_hidden=128, predict_var=False, use_layernorm=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first conv
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels) if use_layernorm else nn.Identity())

        # additional convs
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels) if use_layernorm else nn.Identity())

        self.dropout = dropout
        self.mlp = MLPRegressor(hidden_channels, mlp_hidden=mlp_hidden,
                               dropout=dropout, use_layernorm=use_layernorm,
                               predict_var=predict_var)
        self.predict_var = predict_var
        self.hidden_channels = hidden_channels

    def forward(self, data):
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
        """Get all weights from the entire model"""
        weights = {}
        
        # Get GraphSAGE layer weights
        for i, conv in enumerate(self.convs):
            if hasattr(conv, 'lin_l') and hasattr(conv, 'lin_r'):
                weights[f'conv_{i}_lin_l_weight'] = conv.lin_l.weight.data
                weights[f'conv_{i}_lin_l_bias'] = conv.lin_l.bias.data
                weights[f'conv_{i}_lin_r_weight'] = conv.lin_r.weight.data
                weights[f'conv_{i}_lin_r_bias'] = conv.lin_r.bias.data
        
        # Get MLP weights
        mlp_weights = self.mlp.get_weights()
        for key, value in mlp_weights.items():
            weights[f'mlp_{key}'] = value
            
        return weights
    
    def get_weight_summary(self):
        """Get a summary of all weight shapes and statistics"""
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
        Get predictions for the entire dataset
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            The graph data object
        device : torch.device, optional
            Device to run inference on. If None, uses model's device
            
        Returns:
        --------
        torch.Tensor
            Predictions for all nodes in the dataset
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
        Get predictions separately for training and test sets
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            The graph data object with train_mask and test_mask
        device : torch.device, optional
            Device to run inference on. If None, uses model's device
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'train_pred': predictions for training nodes
            - 'test_pred': predictions for test nodes
            - 'train_true': true values for training nodes
            - 'test_true': true values for test nodes
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
