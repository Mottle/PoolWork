import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class LineNN(nn.Module):
    def __init__(self, channels: int, max_depth: int = 5, dropout: float = 0.5):
        super(LineNN, self).__init__()
        self.channels = channels
        self.max_depth = max_depth
        self.dropout = dropout

        self.line_mlp = nn.Sequential(
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
        out = self.line_mlp(concatenated)
        return out
    
class LineAttention(nn.Module):
    def __init__(self, channels: int, max_depth: int = 5, num_heads: int = 4, dropout: float = 0.5):
        super(LineAttention, self).__init__()
        self.channels = channels
        self.max_depth = max_depth
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.q_linear = nn.Linear(in_features=channels, out_features=channels)
        self.k_linear = nn.Linear(in_features=channels, out_features=channels)
        self.v_linear = nn.Linear(in_features=channels, out_features=channels)
        self.o_linear = nn.Linear(in_features=channels, out_features=channels)

        # self.depth_linear = nn.Linear(in_features=1, out_features=channels)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        remapped = [xi.unsqueeze(1) for xi in x]  # Add sequence dimension
        concatenated = torch.cat(remapped, dim=1)  # Shape: (batch_size, max_depth, channels)

        q = self.q_linear(concatenated)
        k = self.k_linear(concatenated)
        v = self.v_linear(concatenated)
        o = self.attention(q, k, v)
        o = self.o_linear(o)

        out = o.mean(dim=1)  # Aggregate over the sequence dimension
        return out