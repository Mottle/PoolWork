import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse, to_dense_adj

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

    def normalize(self, edge_index, num_nodes, edge_weight=None):
        # 如果没有权重，默认全 1
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        row, col = edge_index
        deg = degree(col, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight
        return edge_index, norm


    def forward(self, x, edge_index, A_phop = None):
        N = x.size(0)

        if A_phop is None:
            A = to_dense_adj(edge_index, max_num_nodes=N)[0]  # 稠密邻接矩阵 [N, N]
            if self.self_loops:
                A = A + torch.eye(N, device=A.device)  # 自环

        outputs = torch.zeros(N, self.linear.out_features, device=x.device)
        d_weight = torch.softmax(self.d, dim=0)

        for p in range(1, self.P + 1):
            if A_phop is None:
                Ap = torch.matrix_power(A, p)
                edge_index_p, edge_weight_p = dense_to_sparse(Ap)
            else:
                edge_index_p, edge_weight_p = A_phop[p - 1]

            # 对称归一化
            edge_index_p, edge_weight_p = self.normalize(edge_index_p, N, edge_weight_p)
            x = self.linear(x)  # [N, out_channels]
            msg = self.propagate(
                edge_index_p, x=x, edge_weight=edge_weight_p, size=(N, N)
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


def compute_all_U(edge_index, num_nodes, K):
    """
    输入:
        edge_index: PyG 格式的边索引 [2, E]
        num_nodes: 节点数
        K: 最大距离

    输出:
        U_list: 一个长度为 K 的列表, 每个元素是 [N, N] 的矩阵,
                表示最短距离恰好为 h 的邻接掩码
    """
    # 1. 转换为稠密邻接矩阵 A
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [N, N]

    # 2. 初始化 R 列表
    R = []
    for h in range(K + 1):
        Ah = torch.matrix_power(A, h)
        Rh = (Ah > 0).int()
        R.append(Rh)

    # 3. 计算所有 U_h
    U_list = []
    for h in range(1, K + 1):
        sum_prev = sum(R[:h])  # R_0 + R_1 + ... + R_{h-1}
        Uh = (R[h] - sum_prev).clamp(min=0, max=1)
        U_list.append(Uh)

    return U_list
