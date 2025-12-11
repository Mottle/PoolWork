import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from .kan import KAN

class KANBasedGINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) with KAN as the aggregation function's MLP.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, kan_kwargs):
        super().__init__(aggr='add') # GIN 使用 'add' 聚合
        
        # KAN 替代 GIN 原有的两层 MLP:
        # KAN 的输入特征维度是 (1 + epsilon) * h_i
        # 在 GIN 中，epsilon * h_i 由权重 W(eps) * h_i 实现。
        # 这里，我们将 KAN 的输入设置为邻居聚合后的特征 (out_channels) + 自身特征 (in_channels)。
        
        # GIN 消息函数 W(h_i) + MLP((1+eps)h_i + sum(h_j))
        # 通常 GIN 的 MLP 接受 (in_channels) 并输出 (out_channels)
        self.kan = KAN(
            in_features=in_channels,      # GIN Conv 的输入维度是聚合后的特征维度
            out_features=out_channels,
            hidden_features=hidden_channels,
            **kan_kwargs
        )
        
        # KAN 不需要单独的 MLP 参数，因为它的计算是自包含的。
        self.initial_eps = 0.0
        self.eps = nn.Parameter(torch.Tensor([self.initial_eps]))

    def forward(self, x, edge_index):
        # 1. 增加自环 (Add self-loops)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 2. 计算度 (Calculate degree for normalization, if needed, but GIN is simple addition)
        # GIN 仅使用 'add' 聚合，无需度归一化，但这里保留度计算的习惯。
        # 实际上，GIN 的 MessagePassing 机制已经处理了聚合。

        # 3. 消息传递 (Start message passing)
        # self.propagate 内部调用 self.message 和 self.update
        out = self.propagate(edge_index, x=x)
        
        # 4. GIN 更新函数 (GIN Update Function)
        # GIN: h_{i}^{(l+1)} = MLP( (1 + \epsilon) h_i^{(l)} + \sum_{j \in N(i)} h_j^{(l)} )
        
        # 4.1. 结合自身特征 (Combine self features with aggregated features)
        # out: 聚合后的特征 \sum_{j \in N(i)} h_j^{(l)}
        # x: 自身特征 h_i^{(l)}
        
        # 这里使用 (1 + \epsilon) h_i^{(l)} + out 作为 KAN 的输入
        out = (1 + self.eps) * x + out

        # 4.2. KAN 应用 (Apply KAN in place of MLP)
        out = self.kan(out)

        return out

    def message(self, x_j):
        # 消息函数 M(h_j) = h_j
        return x_j