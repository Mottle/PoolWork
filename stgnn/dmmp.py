import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
import torch_geometric.nn as pyg_nn

from torch.nn import Parameter, Sigmoid
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Callable, Optional, Tuple, Union
from torch import Tensor

class DMGGNNConv(nn.Module):
    def __init__(
        self,
        channels: int,
        P: int = 3,
    ):
        super(DMGGNNConv, self).__init__()
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
        self.convs = nn.ModuleList([MResGatedGraphConv(channels, channels) for _ in range(P)])

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
            
            # edge_weight_p = edge_weight_p.view(-1, 1).expand(-1, self.channels)

            # edge_index_p, edge_weight_p = signed_symmetric_normalize(
            #     edge_index_p, N, edge_weight_p
            # )

            msg = self.convs[p - 1](x, edge_index_p, edge_weight_p)

            outputs += msg * d_weight[p - 1]

        return outputs + self.hop_bias
    
class MResGatedGraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        act: Optional[Callable] = Sigmoid(),
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.edge_dim = edge_dim
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        edge_dim = edge_dim if edge_dim is not None else 0
        self.lin_key = Linear(in_channels[1] + edge_dim, out_channels)
        self.lin_query = Linear(in_channels[0] + edge_dim, out_channels)
        self.lin_value = Linear(in_channels[0] + edge_dim, out_channels)

        if root_weight:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=False)
        else:
            self.register_parameter('lin_skip', None)

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_weight: Tensor,
        edge_attr: OptTensor = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # In case edge features are not given, we can compute key, query and
        # value tensors in node-level space, which is a bit more efficient:
        if self.edge_dim is None:
            k = self.lin_key(x[1])
            q = self.lin_query(x[0])
            v = self.lin_value(x[0])
        else:
            k, q, v = x[1], x[0], x[0]

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor,
        #                  edge_attr: OptTensor, edge_weight: Tensor)
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_weight=edge_weight, edge_attr=edge_attr)

        if self.root_weight:
            out = out + self.lin_skip(x[1])

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, k_i: Tensor, q_j: Tensor, v_j: Tensor, edge_attr: OptTensor, edge_weight: Tensor) -> Tensor:

        assert (edge_attr is not None) == (self.edge_dim is not None)

        if edge_attr is not None:
            k_i = self.lin_key(torch.cat([k_i, edge_attr], dim=-1))
            q_j = self.lin_query(torch.cat([q_j, edge_attr], dim=-1))
            v_j = self.lin_value(torch.cat([v_j, edge_attr], dim=-1))

        return edge_weight.reshape(-1, 1) * self.act(k_i + q_j) * v_j

    
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