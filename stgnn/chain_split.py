import torch
import torch.nn as nn
import torch.nn.functional as F

class ChainNN(nn.Module):
    def __init__(self, channels: int, max_depth: int = 5, dropout: float = 0.5):
        super(ChainNN, self).__init__()
        self.channels = channels
        self.max_depth = max_depth
        self.dropout = dropout

        self.chain_mlp = nn.Sequential(
            nn.Linear(in_features=channels * max_depth, out_features=channels * max_depth),
            nn.LeakyReLU(),
            nn.Linear(in_features=channels * max_depth, out_features=channels),
            nn.Dropout(p=self.dropout)
        )

        self.depth_linear = nn.Linear(in_features=1, out_features=channels)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        remapped = []
        for idx, xi in enumerate(x):
            if idx == 0:
                alpha = torch.ones(self.channels).to(xi.device)
            else:
                alpha = self.depth_linear(torch.tensor(idx).float().to(xi.device))
                alpha = torch.sigmoid(alpha)
            xi = xi * alpha
            remapped.append(xi)
        concatenated = torch.cat(remapped, dim=1)
        out = self.chain_mlp(concatenated)
        return out