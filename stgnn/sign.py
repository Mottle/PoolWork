import torch
import torch.nn as nn
from torch_geometric.nn import SignedConv
from torch_sparse import SparseTensor


class StackedSIGN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        num_hops=2,
        dropout=0.5,
    ):
        super().__init__()

        self.num_hops = num_hops
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.push_pe = None

        # 多层 SIGNGCNConv
        self.convs = nn.ModuleList()
        self.convs.append(SignedGCN(in_channels, hidden_channels, num_hops))
        for _ in range(num_layers - 2):
            self.convs.append(SignedGCN(hidden_channels, hidden_channels, num_hops))
        self.convs.append(SignedGCN(hidden_channels, out_channels, num_hops))

        # 缓存邻接矩阵
        self.cached_adj = None

    def _precompute(self, x, edge_index):
        """预计算多阶邻接特征 X, A X, A^2 X, ..."""
        N = x.size(0)

        if self.cached_adj is None:
            self.cached_adj = SparseTensor.from_edge_index(
                edge_index, sparse_sizes=(N, N)
            )

        xs = [x]
        cur = x
        for _ in range(self.num_hops):
            cur = self.cached_adj.matmul(cur)
            xs.append(cur)

        return xs  # list: [x0, x1, x2, ...]

    def forward(self, x, edge_index, batch, *args):
        xs = self._precompute(x, edge_index)

        for conv in self.convs[:-1]:
            x = conv(xs)
            x = torch.relu(x)
            x = self.dropout(x)

        # 最后一层不加激活
        x = self.convs[-1](xs)
        return x
