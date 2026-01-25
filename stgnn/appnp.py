import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.nn import global_mean_pool
from torch import nn

class APPNPs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 mlp_layers=3, K=10, alpha=0.1, dropout=0.5, norm=False):
        super().__init__()

        self.norm = norm
        # ----- 3-layer MLP -----
        mlp = []
        last_dim = in_channels
        for _ in range(mlp_layers):
            mlp.append(nn.Linear(last_dim, hidden_channels))
            mlp.append(nn.LeakyReLU())
            mlp.append(nn.Dropout(dropout))
            if self.norm:
                mlp.append(nn.BatchNorm1d(hidden_channels))
            last_dim = hidden_channels

        self.mlp = nn.Sequential(*mlp)

        # ----- APPNP propagation -----
        self.appnp = APPNP(K=K, alpha=alpha, dropout=dropout)

        # ----- Output layer -----
        self.out_lin = nn.Linear(hidden_channels, out_channels)

    # @torch.compile
    def forward(self, x, edge_index, batch=None, *args, **kwargs):
        # 1. MLP feature transformation
        x = self.mlp(x)

        # 2. APPNP propagation
        x = self.appnp(x, edge_index)

        # 3. Graph-level pooling (if needed)
        if batch is not None:
            x = global_mean_pool(x, batch)

        # 4. Output
        # x = self.out_lin(x)
        return x, 0
