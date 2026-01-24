# import torch
# import torch.nn.functional as F
# from torch.nn import Linear
# from torch_geometric.nn import global_mean_pool
# from torch_scatter import scatter_add


# class DirectionalConv(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, edge_index, edge_weight, deg_inv=None):
#         row, col = edge_index
#         msg = x[col] * edge_weight.unsqueeze(-1)
#         out = scatter_add(msg, row, dim=0, dim_size=x.size(0))
#         if deg_inv is not None:
#             out = out * deg_inv.unsqueeze(-1)
#         return out


# class DGN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers=4, dropout=0.5, k_dirs=1):
#         super().__init__()
#         self.dropout = dropout
#         self.k_dirs = k_dirs
#         self.hidden_channels = hidden_channels

#         self.node_encoder = Linear(in_channels, hidden_channels)
#         self.conv = DirectionalConv()

#         self.layers = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             num_channels = 1 + 1 + 2 * k_dirs  # self + avg + dir+ + dir-
#             fusion_lin = Linear(hidden_channels * num_channels, hidden_channels)
#             self.layers.append(fusion_lin)

#     def compute_direction_weights(self, edge_index, pe):
#         row, col = edge_index
#         u_i = pe[row]   # [E, k_dirs]
#         u_j = pe[col]   # [E, k_dirs]
#         diff = u_j - u_i
#         w_up = F.leaky_relu(diff)
#         w_down = F.leaky_relu(-diff)
#         return w_up, w_down

#     def compute_degree_norm(self, edge_index, num_nodes):
#         row, col = edge_index
#         deg = scatter_add(torch.ones(row.size(0), device=row.device),
#                           row, dim=0, dim_size=num_nodes)
#         deg_inv = deg.pow(-1)
#         deg_inv[deg_inv == float('inf')] = 0
#         return deg_inv

#     def forward(self, x, edge_index, batch, pe, *args, **kwargs):
#         x = self.node_encoder(x)

#         deg_inv = self.compute_degree_norm(edge_index, x.size(0))
#         w_up, w_down = self.compute_direction_weights(edge_index, pe)
#         w_avg = torch.ones(edge_index.size(1), device=x.device)

#         for fusion_lin in self.layers:
#             aggregations = []

#             aggregations.append(x)  # self
#             out_avg = self.conv(x, edge_index, w_avg, deg_inv)
#             aggregations.append(out_avg)

#             for d in range(self.k_dirs):
#                 out_up = self.conv(x, edge_index, w_up[:, d], deg_inv)
#                 out_down = self.conv(x, edge_index, w_down[:, d], deg_inv)
#                 aggregations.append(out_up)
#                 aggregations.append(out_down)

#             x_concat = torch.cat(aggregations, dim=-1)
#             x_new = fusion_lin(x_concat)

#             x = x + x_new
#             x = F.leaky_relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         x = global_mean_pool(x, batch)
#         return x, 0


# # import torch
# # import torch.nn.functional as F
# # from torch.nn import Linear, ReLU, LayerNorm
# # from torch_geometric.nn import GCNConv, global_mean_pool
# # import torch_geometric.transforms as T
# # from torch_geometric.data import Data, Batch

# # class DGN(torch.nn.Module):
# #     def __init__(self, in_channels, hidden_channels, num_layers=4, dropout=0.5):
# #         super(DGN, self).__init__()
# #         self.dropout = dropout

# #         self.node_encoder = Linear(in_channels, hidden_channels)
# #         self.layers = torch.nn.ModuleList()
        
# #         for i in range(num_layers):
# #             conv = GCNConv(hidden_channels, hidden_channels)
# #             self.layers.append(conv)


# #     def get_directional_edge_weights(self, edge_index, eig_vecs):
# #         row, col = edge_index
        
# #         u_i = eig_vecs[row, 1] 
# #         u_j = eig_vecs[col, 1]
        
# #         # 计算梯度绝对值
# #         # 注意：这里加一个 1e-5 或者 +1 可以避免权重为0导致断连，
# #         # 但DGN原意就是让梯度大的地方权重更大，所以直接用 abs 也可以。
# #         edge_weight = torch.abs(u_i - u_j)
        
# #         return edge_weight

# #     def forward(self, x, edge_index, batch, pe, *args, **kwargs):
        
# #         edge_weight = self.get_directional_edge_weights(edge_index, pe)

# #         x = self.node_encoder(x)

# #         for layer in self.layers:
# #             x = layer(x, edge_index, edge_weight=edge_weight)
            
# #             x = F.leaky_relu(x)
# #             x = F.dropout(x, p=self.dropout, training=self.training)

# #         x = global_mean_pool(x, batch)
        
# #         return x, 0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter
from torch_geometric.utils import degree

#https://github.com/Saro00/DGN/blob/master/models/pytorch/dgn_layer.py
class FCLayer(nn.Module):
    """简单的全连接层封装，保持与原版一致"""
    def __init__(self, in_size, out_size, activation='LeakyReLU', dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.act = getattr(nn, activation)() if activation != 'none' else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.fc(x)))

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers, mid_activation='ReLU', last_activation='none', dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        if layers <= 1:
            self.layers.append(FCLayer(in_size, out_size, activation=last_activation, dropout=dropout))
        else:
            self.layers.append(FCLayer(in_size, hidden_size, activation=mid_activation, dropout=dropout))
            for _ in range(layers - 2):
                self.layers.append(FCLayer(hidden_size, hidden_size, activation=mid_activation, dropout=dropout))
            self.layers.append(FCLayer(hidden_size, out_size, activation=last_activation, dropout=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 定义 Scalers (PNA 风格)
def scale_identity(x, avg_d=None):
    return x

def scale_amplification(x, avg_d, deg=None):
    if deg is None:
        # 如果没有传入度数，退化为 Identity 或者报错
        return x

    # 确保 avg_d 是 tensor
    if not isinstance(avg_d, torch.Tensor):
        avg_d = torch.tensor(avg_d, dtype=x.dtype, device=x.device)

    # PNA 公式: x * (log(deg + 1) / log(avg_d + 1))
    # deg 的维度是 [N], x 是 [N, F], 需要 unsqueeze
    return x * (torch.log(deg.unsqueeze(-1) + 1) / torch.log(avg_d + 1))

SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    # 可根据需要添加 'attenuation' 等
}

# --- 2. 核心组件: DGNConv (原 DGNTower 的 PyG 化) ---

class DGNConv(MessagePassing):
    def __init__(self, in_features, out_features, aggregators, scalers, avg_d, 
                 pretrans_layers=1, posttrans_layers=1, edge_dim=0):
        """
        DGN Sparse Convolution
        """
        # "add" 聚合在这里只是占位，实际逻辑在 aggregate 中手动控制
        super().__init__(aggr=None, flow='source_to_target') 
        
        self.in_features = in_features
        self.out_features = out_features
        self.avg_d = avg_d
        self.aggregators = aggregators # e.g., ['mean', 'max', 'dir1']
        self.scaler_names = scalers    # e.g., ['identity', 'amplification']
        
        # 1. Pre-transformation MLP (处理 Pairwise 特征)
        # 输入是 cat(x_i, x_j)，所以维度是 2 * in_features
        self.pretrans = MLP(in_size=2 * in_features + edge_dim, 
                            hidden_size=in_features, 
                            out_size=in_features,
                            layers=pretrans_layers, 
                            mid_activation='ReLU', 
                            last_activation='Identity')

        # 2. 计算 Post-transformation 输入维度
        # DGN 逻辑: Concat([Input, Agg_1_Scale_1, Agg_1_Scale_2, ... Agg_N_Scale_M])
        # 聚合器数量 * Scaler数量 * hidden + 原始Input
        num_aggs = len(aggregators)
        num_scalers = len(scalers)
        self.agg_out_dim = num_aggs * num_scalers * in_features
        
        self.posttrans = MLP(in_size=in_features + self.agg_out_dim,
                             hidden_size=out_features,
                             out_size=out_features,
                             layers=posttrans_layers,
                             mid_activation='ReLU',
                             last_activation='Identity')

    def forward(self, x, edge_index, eigvec=None):
        # x: [N, in_features]
        # eigvec: [N, K] (K 是特征向量数量，用于方向聚合)
        
        # 开始消息传递
        # 我们将 eigvec 也传给 propagate，以便在 message 中使用
        out = self.propagate(edge_index, x=x, eigvec=eigvec)
        
        # Post-trans: 拼接原始特征 x (Residual connection) 和 聚合后的特征 out
        out = torch.cat([x, out], dim=-1)
        return self.posttrans(out)

    def message(self, x_i, x_j, eigvec_i, eigvec_j):
        """
        对应原版: h_cat = cat([h_i, h_j]); h_mod = pretrans(h_cat)
        """
        # 1. 构建 Pairwise Feature
        h_cat = torch.cat([x_i, x_j], dim=-1)
        h_mod = self.pretrans(h_cat) # [E, F]

        # 2. 准备不同聚合器的消息
        # 我们需要返回一个列表或字典，包含用于不同聚合的消息
        # 普通聚合: h_mod
        # 方向聚合: h_mod * |eig_i - eig_j|
        
        messages = {}
        messages['base'] = h_mod
        
        # 如果有方向性聚合器，计算方向权重
        # 假设 aggregators 包含 'dir1', 'dir2' 等，对应 eigvec 的第 0, 1 列
        if eigvec_i is not None:
            # 计算简单的方向梯度作为权重: |v_i - v_j|
            # 这里为了通用性，计算前 K 个方向
            # eig_diff shape: [E, K]
            eig_diff = torch.abs(eigvec_i - eigvec_j) 
            messages['dir_weights'] = eig_diff

        return messages

    def aggregate(self, inputs, index, dim_size=None):
        # inputs 是 message 返回的字典
        h_mod = inputs['base']      # [E, F]
        dir_w = inputs.get('dir_weights', None) # [E, K]
        
        aggr_outs = []

        if dim_size is None:
         dim_size = int(index.max()) + 1

        # 计算度数: deg[i] 表示节点 i 有多少个邻居
        deg = degree(index, dim_size, dtype=inputs['base'].dtype)
        
        for agg_name in self.aggregators:
            if agg_name == 'mean':
                out = scatter(h_mod, index, dim=0, dim_size=dim_size, reduce='mean')
            elif agg_name == 'max':
                out = scatter(h_mod, index, dim=0, dim_size=dim_size, reduce='max')
            elif agg_name == 'min':
                out = scatter(h_mod, index, dim=0, dim_size=dim_size, reduce='min')
            elif agg_name == 'sum':
                out = scatter(h_mod, index, dim=0, dim_size=dim_size, reduce='sum')
            
            # --- 方向性聚合 ---
            elif agg_name.startswith('dir'):
                # 假设格式是 dir0, dir1, dir2... 代表使用第 k 个特征向量
                try:
                    k = int(agg_name.replace('dir', ''))
                    if dir_w is not None and k < dir_w.size(1):
                        # Weight the message: h_mod * w_k
                        # w_k: [E], h_mod: [E, F] -> 广播乘法
                        w_k = dir_w[:, k].unsqueeze(-1)
                        msg_dir = h_mod * w_k
                        # DGN 原文通常对方向性聚合使用 'mean' 或 'sum'，这里默认 'sum' (积分)
                        out = scatter(msg_dir, index, dim=0, dim_size=dim_size, reduce='mean')
                    else:
                        # Fallback if eigvec missing
                        out = torch.zeros_like(h_mod[:dim_size])
                except:
                    out = torch.zeros_like(h_mod[:dim_size])
            else:
                raise ValueError(f"Unknown aggregator: {agg_name}")
            
            aggr_outs.append(out)

        # 此时 aggr_outs 是一个 list，包含 [N, F], [N, F]...
        # 接下来应用 Scalers (对应原版 loop over scalers)
        
        scaled_outs = []
        for out in aggr_outs:
            for scale_name in self.scaler_names:
                # 传入 deg
                if scale_name == 'amplification':
                    s_out = scale_amplification(out, avg_d=self.avg_d, deg=deg)
                elif scale_name == 'identity':
                    s_out = scale_identity(out)
                else:
                    s_out = out # Fallback
                scaled_outs.append(s_out)
        
        # 最终拼接: [N, Num_Agg * Num_Scale * F]
        return torch.cat(scaled_outs, dim=-1)

# --- 3. 封装层: DGNLayer (多塔管理) ---

class DGNLayer(nn.Module):
    def __init__(self, in_features, out_features, aggregators, scalers, avg_d, 
                 towers=1, pretrans_layers=1, posttrans_layers=1, divide_input=True):
        super().__init__()
        
        # 逻辑检查
        if divide_input:
            assert in_features % towers == 0
            tower_in = in_features // towers
        else:
            tower_in = in_features
            
        assert out_features % towers == 0
        tower_out = out_features // towers
        
        self.divide_input = divide_input
        self.input_tower = tower_in
        self.towers = nn.ModuleList()
        
        # 构建 Towers (即多个 DGNConv)
        for _ in range(towers):
            self.towers.append(
                DGNConv(in_features=tower_in, 
                        out_features=tower_out, 
                        aggregators=aggregators,
                        scalers=scalers,
                        avg_d=avg_d,
                        pretrans_layers=pretrans_layers,
                        posttrans_layers=posttrans_layers)
            )
            
        # Mixing Network (对应原版 mixing_network)
        self.mixing_network = FCLayer(out_features, out_features, activation='LeakyReLU')

    def forward(self, x, edge_index, eigvec=None):
        tower_outs = []
        for i, tower in enumerate(self.towers):
            # Input Slicing
            if self.divide_input:
                start = i * self.input_tower
                end = (i + 1) * self.input_tower
                x_tower = x[:, start:end]
            else:
                x_tower = x
                
            # Tower Forward
            out = tower(x_tower, edge_index, eigvec)
            tower_outs.append(out)
            
        # Cat & Mix
        y = torch.cat(tower_outs, dim=-1)
        return self.mixing_network(y)

# --- 4. 最终模型: DGN Graph Classifier ---

class DGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, 
                 aggregators=['mean', 'max', 'dir0', 'dir1'], 
                 scalers=['identity', 'amplification'], 
                 avg_d=2.0, towers=4, dropout=0.5):
        super().__init__()
        
        self.dropout = dropout
        
        # 1. Embedding
        self.embedding = nn.Linear(in_channels, hidden_channels)
        
        # 2. Backbone (DGN Layers)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                DGNLayer(in_features=hidden_channels,
                         out_features=hidden_channels,
                         aggregators=aggregators,
                         scalers=scalers,
                         avg_d=avg_d,
                         towers=towers,
                         divide_input=True) # 通常 DGN 使用 divide_input=True
            )

    def forward(self, x, edge_index, batch, eigvec=None, *args, **kwargs):
        # 预处理 Eigenvectors (如果传入了)
        # 如果 Dataset 没有提供 eigvec，代码会自动处理为全0的 fallback
        
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x, edge_index, eigvec)
            # DGNLayer 内部已经包含了 activation (Mixing Net 有 LeakyReLU)
            # 所以这里主要加 Dropout 和 Residual (可选)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Graph Pooling
        x = global_mean_pool(x, batch)
        
        return x, 0