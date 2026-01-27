from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse, to_dense_adj
from torch_scatter import scatter_max, scatter_add, scatter_softmax
from torch_geometric.utils import coalesce
import math

# from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_scatter

# class PHopGCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, p=1, aggr='add'):
#         super(PHopGCNConv, self).__init__(aggr=aggr)
#         self.p = p
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#         # 初始化为全 1
#         self.hop_params = torch.nn.Parameter(torch.ones(p, out_channels))

#     def forward(self, x, edge_index):
#         # 加自环
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # 计算度数
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)

#         # 对称归一化
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         hop_weights = torch.softmax(self.hop_params, dim=0)  # [p, out_channels]

#         h = x
#         out_list = []
#         for hop in range(self.p):
#             h = self.propagate(edge_index, x=h, norm=norm)   # 一次 1-hop
#             weighted_h = h * hop_weights[hop]                # 通道级加权
#             out_list.append(weighted_h)

#         out = torch.stack(out_list, dim=0).sum(dim=0)        # 累加所有 hop
#         return self.lin(out)

#     def message(self, x_j, norm):
#         # GCN 的消息函数：邻居特征乘归一化系数
#         return norm.view(-1, 1) * x_j

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class PHopGCNConv(MessagePassing):
    def __init__(
        self, in_channels: int, out_channels: int, p: int = 1, aggr: str = "add"
    ):
        super(PHopGCNConv, self).__init__(aggr=aggr)
        self.p = p
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        # p 个通道级参数，初始化为全 1
        self.hop_params = torch.nn.Parameter(torch.ones(p, out_channels))
        # p 个 bias，每个 hop 一个向量
        self.hop_bias = torch.nn.Parameter(torch.zeros(p, out_channels))

    def forward(self, x, edge_index):
        # 加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        hop_weights = torch.softmax(self.hop_params, dim=0)  # [p, out_channels]

        h = x
        out_list = []
        for hop in range(self.p):
            h = self.propagate(edge_index, x=h, norm=norm)  # 一次 1-hop
            weighted_h = h * hop_weights[hop] + self.hop_bias[hop]  # 每跳加 bias
            out_list.append(weighted_h)

        out = torch.stack(out_list, dim=0).sum(dim=0)  # 累加所有 hop
        out = self.lin(out)  # 线性变换
        return out

    def message(self, x_j, norm):
        # GCN 的消息函数：邻居特征乘归一化系数
        return norm.view(-1, 1) * x_j


class PHopLinkGCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        P: int = 3,
        aggr: str = "add",
        self_loops: bool = True,
    ):
        super(PHopLinkGCNConv, self).__init__(aggr=aggr)
        self.P = P
        self.self_loops = self_loops
        self.d = nn.Parameter(torch.ones(P, out_channels))
        self.hop_bias = nn.Parameter(torch.zeros(P, out_channels))
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, A_phop=None):
        N = x.size(0)

        if A_phop is None:
            raise ValueError("A_phop is None")

        outputs = torch.zeros(N, self.linear.out_features, device=x.device)
        d_weight = torch.softmax(self.d, dim=0)

        xp = self.linear(x)
        edge_index_l, edge_wight_l = A_phop

        for p in range(1, self.P + 1):
            edge_index_p, edge_weight_p = edge_index_l[p - 1], edge_wight_l[p - 1]

            if self.self_loops:
                edge_index_p, edge_weight_p = add_self_loops(
                    edge_index_p, edge_weight_p, fill_value=1, num_nodes=N
                )

            # edge_index_p, edge_weight_p = signed_symmetric_normalize(
            #     edge_index_p, N, edge_weight_p
            # )
            edge_index_p, edge_weight_p = symmetric_normalize(
                edge_index_p, N, edge_weight_p
            )


            # xp = self.linear(x)  # [N, out_channels]
            msg = self.propagate(
                edge_index_p, x=xp, edge_weight=edge_weight_p, size=(N, N)
            )
            # msg = self.linear(msg)  # [N, out_channels] fix

            # outputs += msg * self.d[p-1] + self.hop_bias[p-1]
            outputs += msg * d_weight[p - 1] + self.hop_bias[p - 1]

        # return F.relu(outputs)
        return outputs

    def message(self, x_j, edge_weight):
        # 消息 = 邻居特征 * 路径数量
        return edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(
            inputs, index, dim=0, reduce=self.aggr, dim_size=dim_size
        )


class PHopLinkRWConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        P: int = 3,
        aggr: str = "add",
        # self_loops: bool = True,
    ):
        super(PHopLinkRWConv, self).__init__(aggr=aggr)
        self.P = P
        # self.self_loops = self_loops
        self.d = nn.Parameter(torch.ones(P, out_channels))
        self.hop_bias = nn.Parameter(torch.zeros(P, out_channels))
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, A_phop=None):
        N = x.size(0)

        if A_phop is None:
            raise ValueError("A_phop is None")

        outputs = torch.zeros(N, self.linear.out_features, device=x.device)
        d_weight = torch.softmax(self.d, dim=0)

        xp = self.linear(x)
        edge_index_l, edge_wight_l = A_phop

        for p in range(1, self.P + 1):
            # edge_index_p, edge_weight_p = A_phop[p - 1]
            edge_index_p, edge_weight_p = edge_index_l[p - 1], edge_wight_l[p - 1]

            # 对称归一化
            # edge_index_p, edge_weight_p = symmetric_normalize(
            #     edge_index_p, N, edge_weight_p
            # )

            # 随机游走归一化
            edge_index_p, edge_weight_p = random_walk_normalize(
                edge_index_p, N, edge_weight_p, smoothing=False
            )

            # xp = self.linear(x)  # [N, out_channels]
            # propagate = torch.compile(self.propagate)
            msg = self.propagate(
                edge_index_p, x=xp, edge_weight=edge_weight_p, size=(N, N)
            )
            # msg = self.linear(msg)  # [N, out_channels] fix

            # 每个 hop 的向量权重和偏置
            # outputs += msg * self.d[p-1] + self.hop_bias[p-1]
            outputs += msg * d_weight[p - 1] + self.hop_bias[p - 1]

        # return F.relu(outputs)
        return outputs

    def message(self, x_j, edge_weight):
        # 消息 = 邻居特征 * 路径数量
        return edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(
            inputs, index, dim=0, reduce=self.aggr, dim_size=dim_size
        )


class PHopLinkGINRWConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        P: int = 3,
        aggr: str = "add",
        # add_self_loops: bool = False,
    ):
        super(PHopLinkGINRWConv, self).__init__(aggr=aggr)
        self.P = P
        # self.add_self_loops = add_self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.eps = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, out_channels),
        )

        # 多跳权重
        self.d = nn.Parameter(torch.ones(P, out_channels))
        self.hop_bias = nn.Parameter(torch.zeros(out_channels))
        # self.hop_linear = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(P)])

    def forward(self, x, edge_index, A_phop=None):
        N = x.size(0)

        if A_phop is None:
            raise ValueError("A_phop is None")

        outputs = torch.zeros(N, self.out_channels, device=x.device)
        d_weight = torch.softmax(self.d, dim=0)

        edge_index_l, edge_wight_l = A_phop

        for p in range(1, self.P + 1):
            edge_index_p, edge_weight_p = edge_index_l[p - 1], edge_wight_l[p - 1]

            # 随机游走归一化
            edge_index_p, edge_weight_p = random_walk_normalize(
                edge_index_p, N, edge_weight_p, smoothing=False
            )

            msg = self.propagate(
                edge_index_p, x=x, edge_weight=edge_weight_p, size=(N, N)
            )

            msg = msg + (1 + self.eps) * x

            # outputs += msg * d_weight[p - 1] + self.hop_bias[p - 1]
            # outputs += self.hop_linear[p - 1](msg)
            outputs += self.mlp(msg) * d_weight[p - 1]

        # outputs = self.mlp(outputs)

        return outputs + self.hop_bias

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(
            inputs, index, dim=0, reduce=self.aggr, dim_size=dim_size
        )
    
class PHopLinkGINDiffConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        P: int = 3,
        aggr: str = "add",
    ):
        super(PHopLinkGINDiffConv, self).__init__(aggr=aggr)
        self.P = P
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.eps = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, out_channels),
        )

        # 多跳权重
        self.d = nn.Parameter(torch.ones(P, out_channels))
        self.hop_bias = nn.Parameter(torch.zeros(out_channels))
        # self.hop_linear = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(P)])

    def forward(self, x, edge_index, A_phop=None):
        N = x.size(0)

        if A_phop is None:
            raise ValueError("A_phop is None")

        outputs = torch.zeros(N, self.out_channels, device=x.device)
        d_weight = torch.softmax(self.d, dim=0)

        edge_index_l, edge_wight_l = A_phop

        for p in range(1, self.P + 1):
            edge_index_p, edge_weight_p = edge_index_l[p - 1], edge_wight_l[p - 1]

            # 随机游走归一化
            edge_index_p, edge_weight_p = random_walk_normalize(
                edge_index_p, N, edge_weight_p, smoothing=False
            )

            msg = self.propagate(
                edge_index_p, x=x, edge_weight=edge_weight_p, size=(N, N)
            )

            msg = msg + (1 + self.eps) * x

            # outputs += msg * d_weight[p - 1] + self.hop_bias[p - 1]
            # outputs += self.hop_linear[p - 1](msg)
            outputs += self.mlp(msg) * d_weight[p - 1]

        # outputs = self.mlp(outputs)

        return outputs + self.hop_bias

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(
            inputs, index, dim=0, reduce=self.aggr, dim_size=dim_size
        )


class PHopLinkGINConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        P: int = 3,
        aggr: str = "add",
        norm: Optional[str] = 'signed_sym'
    ):
        super(PHopLinkGINConv, self).__init__(aggr=aggr)
        self.P = P
        # self.self_loops = self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.eps = nn.Parameter(torch.zeros(1))

        self.mlp = nn.ModuleList([self.build_mlp(in_channels, out_channels) for _ in range(P)])
        self.norm = norm

        # 多跳权重
        self.d = nn.Parameter(torch.ones(P, out_channels))
        self.hop_bias = nn.Parameter(torch.zeros(out_channels))
        # self.hop_linear = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(P)])
    
    def build_mlp(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, A_phop=None):
        N = x.size(0)

        if A_phop is None:
            raise ValueError("A_phop is None")

        outputs = torch.zeros(N, self.out_channels, device=x.device)
        d_weight = torch.softmax(self.d, dim=0)

        edge_index_l, edge_wight_l = A_phop

        for p in range(1, self.P + 1):
            edge_index_p, edge_weight_p = edge_index_l[p - 1], edge_wight_l[p - 1]

            # if self.self_loops:
            #     edge_index_p, edge_weight_p = add_self_loops(
            #         edge_index_p, num_nodes=N, edge_attr=edge_weight_p
            #     )
            
            if self.norm is not None and p != 1:
                # edge_index_p, edge_weight_p = diffusion_normalize(edge_index_p, edge_weight_p, num_nodes=N)
                if self.norm == 'signed_sym':
                    edge_index_p, edge_weight_p = signed_symmetric_normalize(edge_index_p, N, edge_weight_p)
                elif self.norm == 'sym':
                    edge_index_p, edge_weight_p = symmetric_normalize(edge_index_p, N, edge_weight_p)

            msg = self.propagate(
                edge_index_p, x=x, edge_weight=edge_weight_p, size=(N, N)
            )

            msg = msg + (1 + self.eps) * x
            outputs += self.mlp[p-1](msg) * d_weight[p - 1]

        return outputs + self.hop_bias

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(
            inputs, index, dim=0, reduce=self.aggr, dim_size=dim_size
        )


class PHopGINConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        p: int = 2,
        aggr: str = "add",
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        super(PHopGINConv, self).__init__(aggr=aggr)
        self.p = p
        # GIN 使用 MLP，这里用两层线性+ReLU
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            # torch.nn.ReLU(),
            # torch.nn.Linear(out_channels, out_channels)
        )
        # p 个通道级参数，初始化为全 1
        self.hop_params = torch.nn.Parameter(torch.ones(p, out_channels))
        # p 个 bias，每个 hop 一个向量
        self.hop_bias = torch.nn.Parameter(torch.zeros(p, out_channels))
        # epsilon 参数
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        hop_weights = torch.softmax(self.hop_params, dim=0)  # [p, out_channels]

        h = x
        out_list = []
        for hop in range(self.p):
            h = self.propagate(edge_index, x=h)  # 一次 1-hop
            # GIN 的 (1+eps)*x + sum(neighbors)
            h = (1 + self.eps) * x + h
            weighted_h = h * hop_weights[hop] + self.hop_bias[hop]
            out_list.append(weighted_h)

        out = torch.stack(out_list, dim=0).sum(dim=0)  # 累加所有 hop
        out = self.mlp(out)  # MLP
        return out

    def message(self, x_j):
        return x_j


########WRONG########
# def random_walk_normalize(edge_index, num_nodes, edge_weight=None, smoothing=False):
#     if edge_weight is None:
#         edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

#     row, col = edge_index
#     deg = degree(row, num_nodes, dtype=edge_weight.dtype)  # 出度（源节点的度）

#     # 避免除零错误
#     deg_inv = deg.pow(-1)
#     deg_inv[deg_inv == float("inf")] = 0

#     # 随机游走概率归一化
#     norm = deg_inv[row] * edge_weight

#     if smoothing:
#         norm = grouped_softmax(edge_index, norm, num_nodes)
#         # norm_out = torch.zeros_like(norm)
#         # for i in range(num_nodes):
#         #     mask = (row == 1)
#         #     if mask.sum() > 0:
#         #         norm_out[i] = torch.softmax(norm[mask], dim=0)
#         # norm = norm_out

#     return edge_index, norm

# def random_walk_normalize(edge_index, num_nodes, edge_weight=None, smoothing=False):
#     if edge_weight is None:
#         edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

#     row, col = edge_index

#     # 计算出度
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

#     # RW 归一化
#     norm = edge_weight / (deg[row] + 1e-16)

#     # 可选 smoothing（grouped softmax）
#     if smoothing:
#         norm = scatter_softmax(norm, row, dim=0)

#     return edge_index, norm
########WRONG######## END


# @torch.compile
def random_walk_normalize(edge_index, num_nodes, edge_weight=None, smoothing=False):
    # if edge_weight is None:
    #     edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    row, col = edge_index

    if smoothing:
        # 直接对 U^{(p)}_{ij} 做邻域 softmax
        norm = scatter_softmax(edge_weight, row, dim=0)
    else:
        # 传统 RW 归一化
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        norm = edge_weight / (deg[row] + 1e-16)

    return edge_index, norm


# def grouped_softmax(edge_index, edge_weight, num_nodes):
#     row, col = edge_index
#     # 计算每个节点的最大值（用于 softmax 的稳定性）
#     max_per_row, _ = scatter_max(edge_weight, row, dim=0, dim_size=num_nodes)
#     max_per_row = max_per_row[row]

#     # 减去最大值，避免溢出
#     exp_weight = torch.exp(edge_weight - max_per_row)

#     # 计算分母（每个节点的 exp 和）
#     sum_per_row = scatter_add(exp_weight, row, dim=0, dim_size=num_nodes)
#     sum_per_row = sum_per_row[row]

#     # softmax 结果
#     norm_out = exp_weight / (sum_per_row + 1e-16)
#     return norm_out


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


def diffusion_normalize(powers_index, powers_weight, method='ppr', alpha=0.1, t=2.0, num_nodes=None):
    """
    参数:
    - powers_index: List[Tensor], 每一项是 A^k 的 edge_index
    - powers_weight: List[Tensor], 每一项是 A^k 的 edge_weight
    - method: 'ppr' 或 'heat'
    - alpha: PPR 的重启概率 (0 < alpha < 1)
    - t: Heat Kernel 的扩散时间
    - num_nodes: 图的节点总数
    """
    # 健壮性检查：如果传入的是单个 Tensor 而不是列表，将其包装成列表
    if isinstance(powers_index, torch.Tensor):
        powers_index = [powers_index]
    if isinstance(powers_weight, torch.Tensor):
        powers_weight = [powers_weight]

    K = len(powers_index) - 1
    all_indices = []
    all_weights = []

    for k in range(K + 1):
        # 计算权重 w_k
        if method == 'ppr':
            weight_k = alpha * (float(1 - alpha) ** k)
        elif method == 'heat':
            weight_k = math.exp(-t) * (t ** k) / math.factorial(k)
        
        # 核心修复：确保 index 是 [2, E] 形状且 weight 是 [E] 形状
        idx = powers_index[k]
        w = powers_weight[k]
        
        # 自动处理可能存在的空边情况
        if idx.numel() == 0:
            continue
            
        all_indices.append(idx)
        all_weights.append(w * weight_k)

    # 如果列表为空（没有任何边），返回空的 sparse 结构
    if len(all_indices) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=powers_index[0].device), \
               torch.zeros((0,), device=powers_index[0].device)

    # 1. 拼接
    final_edge_index = torch.cat(all_indices, dim=1)  # 这里不会再报错，因为已确保是 Tensor 列表
    final_edge_weight = torch.cat(all_weights, dim=0)

    # 2. 合并重复项 (coalesce)
    edge_index_diff, edge_weight_diff = coalesce(
        final_edge_index, 
        final_edge_weight, 
        num_nodes=num_nodes
    )

    return edge_index_diff, edge_weight_diff