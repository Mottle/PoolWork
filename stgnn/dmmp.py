import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
import torch_geometric.nn as pyg_nn


class DMGatedGCNConv(nn.Module):
    def __init__(
        self,
        channels: int,
        P: int = 3,
    ):
        super(DMGatedGCNConv, self).__init__()
        self.P = P
        self.channels = channels
        self.d = nn.Parameter(torch.ones(P, channels))
        self.hop_bias = nn.Parameter(torch.zeros(channels))
        self.convs = nn.ModuleList([pyg_nn.GatedGraphConv(channels, 1) for _ in range(P)])

    def forward(self, x, edge_index, edge_attr = None, A_phop=None):
        N = x.size(0)

        if A_phop is None:
            raise ValueError("A_phop is None")

        outputs = torch.zeros(N, self.channels, device=x.device)
        d_weight = torch.softmax(self.d, dim=0)

        edge_index_l, edge_wight_l = A_phop

        for p in range(1, self.P + 1):
            edge_index_p, edge_weight_p = edge_index_l[p - 1], edge_wight_l[p - 1]
            edge_index_p, edge_weight_p = symmetric_normalize(
                edge_index_p, N, edge_weight_p
            )
            if p == 1 and edge_attr is not None:
                edge_weight_p = edge_attr * edge_weight_p

            # edge_index_p, edge_weight_p = signed_symmetric_normalize(
            #     edge_index_p, N, edge_weight_p
            # )

            msg = self.convs[p - 1](x, edge_index_p, edge_weight_p)

            outputs += msg * d_weight[p - 1]

        return outputs + self.hop_bias

    
def symmetric_normalize(edge_index, num_nodes, edge_weight=None):
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=edge_weight.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight
    return edge_index, norm


def signed_symmetric_normalize(edge_index, num_nodes, edge_weight):
    # 1. 确保 edge_weight 是 tensor 且是 float
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    if isinstance(edge_weight, (int, float)):
        edge_weight = torch.full((edge_index.size(1),), edge_weight, device=edge_index.device)
    if not edge_weight.is_floating_point():
        edge_weight = edge_weight.to(torch.float32)

    # 2. 确保 num_nodes 是 int (防止之前的报错)
    if isinstance(num_nodes, torch.Tensor):
        num_nodes = int(num_nodes.item())
    num_nodes = int(num_nodes) if num_nodes is not None else int(edge_index.max()) + 1

    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg.scatter_add_(0, col, edge_weight.abs()) 
    
    # 3. 计算 D^-0.5
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight
    
    return edge_index, norm