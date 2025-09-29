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
