import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer.

    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph = full_graph

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        if self.full_graph:
            self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.fake_edge_emb = fake_edge_emb

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        if self.full_graph:
            fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)
            src_2 = batch.K_2h[fake_edge_index[0]]  # (num fake edges) x num_heads x out_dim
            dest_2 = batch.Q_2h[fake_edge_index[1]]  # (num fake edges) x num_heads x out_dim
            score_2 = torch.mul(src_2, dest_2)

            # Scale scores by sqrt(d)
            score_2 = score_2 / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim

        if self.full_graph:
            # E_2 is 1 x num_heads x out_dim and will be broadcast over dim=0
            score_2 = torch.mul(score_2, batch.E_2)

        if self.full_graph:
            # softmax and scaling by gamma
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1
            score_2 = torch.exp(score_2.sum(-1, keepdim=True).clamp(-5, 5))  # (num fake edges) x num_heads x 1
            score = score / (self.gamma + 1)
            score_2 = self.gamma * score_2 / (self.gamma + 1)
        else:
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]

        #MODIFIED CODE HERE #############
        ###################################################
        # batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        # scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')
        ###################################################
        batch.wV = scatter(msg, batch.edge_index[1], dim=0, dim_size=batch.V_h.size(0), reduce='add')

        if self.full_graph:
            # Attention via fictional edges
            msg_2 = batch.V_h[fake_edge_index[0]] * score_2
            # Add messages along fake edges to destination nodes
            scatter(msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, batch.edge_index[1], dim=0, out=batch.Z, reduce='add')
        if self.full_graph:
            scatter(score_2, fake_edge_index[1], dim=0, out=batch.Z, reduce='add')

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        E = self.E(batch.edge_attr)

        if self.full_graph:
            Q_2h = self.Q_2(batch.x)
            K_2h = self.K_2(batch.x)
            # One embedding used for all fake edges; shape: 1 x emb_dim
            dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
            E_2 = self.E_2(dummy_edge)

        V_h = self.V(batch.x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
            batch.K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
            batch.E_2 = E_2.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)

        h_out = batch.wV / (batch.Z + 1e-6)

        return h_out


class SANLayer(nn.Module):
    """GraphTransformerLayer from SAN.

    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = MultiHeadAttentionLayer(gamma=gamma,
                                                 in_dim=in_dim,
                                                 out_dim=out_dim // num_heads,
                                                 num_heads=num_heads,
                                                 full_graph=full_graph,
                                                 fake_edge_emb=fake_edge_emb,
                                                 use_bias=use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)


def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short,
                           device=edge_index.device)
        adj = scatter(zero, idx, dim=0, dim_size=flattened_size, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative



class SANBatchData:
    """
    一个轻量级的容器，用于模拟 PyG Batch 对象的行为。
    它允许 SANLayer 通过 .x, .edge_index 等属性访问数据，
    并支持动态添加属性 (如 Q_h, wV 等)。
    """
    def __init__(self, x, edge_index, batch, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.edge_attr = edge_attr
        
        # 初始化 SANLayer 可能需要的中间变量属性，防止 AttributeError
        self.Q_h = None
        self.K_h = None
        self.V_h = None
        self.E = None
        self.wV = None
        self.Z = None

    def size(self, dim):
        """
        兼容性修复：
        部分 SAN 实现中可能调用 batch.size(0) 来获取节点数。
        PyG 的 Data 对象通常没有 size() 方法，但为了防止报错，
        这里将其代理给 x 的 size。
        """
        return self.x.size(dim)



class SAN(nn.Module):
    def __init__(self, 
                 in_dim, 
                 edge_dim, 
                 hidden_dim, 
                 pe_dim=20,
                 num_layers=4, 
                 num_heads=4, 
                 gamma=0.0,
                 full_graph=True, 
                 dropout=0.0, 
                 residual=True,
                 layer_norm=False,
                 batch_norm=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.edge_dim = edge_dim
        # 1. 节点特征编码
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        
        # 2. 边特征编码
        if edge_dim > 0:
            self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_encoder = nn.Linear(1, hidden_dim) 

        self.pe_encoder = nn.Linear(pe_dim, hidden_dim)
            
        # 3. 虚拟边嵌入
        self.fake_edge_emb = nn.Embedding(1, hidden_dim)
        nn.init.xavier_uniform_(self.fake_edge_emb.weight)

        # 4. 堆叠 SAN Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                SANLayer(
                    gamma=gamma,
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    full_graph=full_graph,
                    fake_edge_emb=self.fake_edge_emb,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    batch_norm=batch_norm,
                    residual=residual
                )
            )

    def forward(self, x, edge_index, batch, pe, edge_attr=None, *args, **kwargs):
        """
        参数改为解耦形式：
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]
        batch: [num_nodes]
        edge_attr: [num_edges, edge_dim] (可选)
        """
        
        # 编码节点
        h = self.node_encoder(x)
        pe_emb = self.pe_encoder(pe)
        h = h + pe_emb
        
        # 编码边
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_emb = self.edge_encoder(edge_attr.float())
        else:
            # 如果没有提供 edge_attr，生成 dummy embedding
            num_edges = edge_index.size(1)
            dummy_attr = h.new_zeros(num_edges, self.edge_dim)
            edge_emb = self.edge_encoder(dummy_attr)

        batch_data = SANBatchData(h, edge_index, batch, edge_emb)

        # --- 3. 层传播 ---
        for layer in self.layers:
            # layer 会就地修改 batch_data.x
            batch_data = layer(batch_data)
        
        # --- 4. 全局池化与输出 ---
        # 从封装对象中取出最终的节点特征 x
        graph_emb = global_mean_pool(batch_data.x, batch_data.batch)
        
        return graph_emb, 0


# import torch
# from torch import nn
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.nn import global_mean_pool


# class SignNet(nn.Module):
#     """
#     向量版 SignNet：
#     pe: [N, K] -> [N, H]
#     f(x) + f(-x) 保证符号不变性
#     """
#     def __init__(self, pe_dim: int, hidden_channels: int):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(pe_dim, hidden_channels),
#             nn.PReLU(),
#             nn.Linear(hidden_channels, hidden_channels),
#         )

#     def forward(self, pe: torch.Tensor) -> torch.Tensor:
#         if pe is None:
#             return None
#         out_pos = self.mlp(pe)      # [N, H]
#         out_neg = self.mlp(-pe)     # [N, H]
#         return out_pos + out_neg    # [N, H]


# class SANStructuralBias(nn.Module):
#     """
#     完整 SAN 结构偏置（严格对齐原文）：
#     - head-wise bias: [B, heads, N, N]
#     - LPE difference bias
#     - SPD bias（可选）
#     - degree bias（centrality）
#     - edge bias（可选）
#     """
#     def __init__(self, pe_dim, hidden_channels, heads, use_spd=False, use_degree=True, use_edge=False):
#         super().__init__()
#         self.heads = heads
#         self.use_spd = use_spd
#         self.use_degree = use_degree
#         self.use_edge = use_edge

#         # LPE difference bias → head-wise
#         self.pe_mlp = nn.Sequential(
#             nn.Linear(pe_dim, hidden_channels),
#             nn.PReLU(),
#             nn.Linear(hidden_channels, heads)  # 每个 head 一个 bias
#         )

#         # SPD bias
#         if use_spd:
#             self.spd_mlp = nn.Sequential(
#                 nn.Linear(1, hidden_channels),
#                 nn.PReLU(),
#                 nn.Linear(hidden_channels, heads)
#             )
#         else:
#             self.spd_mlp = None

#         # degree bias（centrality）
#         if use_degree:
#             self.degree_emb = nn.Embedding(512, heads)  # 假设度 < 512
#         else:
#             self.degree_emb = None

#         # edge bias（可选）
#         if use_edge:
#             self.edge_mlp = nn.Sequential(
#                 nn.Linear(1, hidden_channels),
#                 nn.PReLU(),
#                 nn.Linear(hidden_channels, heads)
#             )
#         else:
#             self.edge_mlp = None

#     def forward(self, pe_dense, degree_dense=None, spd_dense=None, edge_dense=None):
#         """
#         pe_dense:     [B, N, K]
#         degree_dense: [B, N]
#         spd_dense:    [B, N, N]
#         edge_dense:   [B, N, N]
#         return:       [B, heads, N, N]
#         """
#         B, N, K = pe_dense.size()

#         # -------------------------
#         # 1. LPE difference bias
#         # -------------------------
#         pe_i = pe_dense.unsqueeze(2)          # [B, N, 1, K]
#         pe_j = pe_dense.unsqueeze(1)          # [B, 1, N, K]
#         pe_diff = torch.abs(pe_i - pe_j)      # [B, N, N, K]

#         pe_bias = self.pe_mlp(pe_diff)        # [B, N, N, heads]

#         bias = pe_bias

#         # -------------------------
#         # 2. SPD bias
#         # -------------------------
#         if self.use_spd and spd_dense is not None:
#             spd_feat = self.spd_mlp(spd_dense.unsqueeze(-1))  # [B, N, N, heads]
#             bias = bias + spd_feat

#         # -------------------------
#         # 3. degree bias
#         # -------------------------
#         if self.use_degree and degree_dense is not None:
#             deg_emb = self.degree_emb(degree_dense)  # [B, N, heads]
#             deg_i = deg_emb.unsqueeze(2)             # [B, N, 1, heads]
#             deg_j = deg_emb.unsqueeze(1)             # [B, 1, N, heads]
#             bias = bias + (deg_i + deg_j)

#         # -------------------------
#         # 4. edge bias
#         # -------------------------
#         if self.use_edge and edge_dense is not None:
#             edge_feat = self.edge_mlp(edge_dense.unsqueeze(-1))  # [B, N, N, heads]
#             bias = bias + edge_feat

#         # -------------------------
#         # 5. reshape to [B, heads, N, N]
#         # -------------------------
#         bias = bias.permute(0, 3, 1, 2).contiguous()

#         return bias



# class SANTransformerLayer(nn.Module):
#     def __init__(self, hidden_channels, heads, dropout):
#         super().__init__()
#         self.heads = heads

#         self.attn = nn.MultiheadAttention(
#             embed_dim=hidden_channels,
#             num_heads=heads,
#             dropout=dropout,
#             batch_first=True
#         )

#         self.norm1 = nn.LayerNorm(hidden_channels)
#         self.ff = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_channels * 2, hidden_channels),
#             nn.Dropout(dropout),
#         )
#         self.norm2 = nn.LayerNorm(hidden_channels)

#     def forward(self, x, attn_bias, key_padding_mask):
#         """
#         x:         [B, N, H]
#         attn_bias: [B, heads, N, N]
#         """
#         B, N, H = x.size()

#         # MultiheadAttention 需要 [B*heads, N, N]
#         attn_mask = attn_bias.reshape(B * self.heads, N, N)

#         x2, _ = self.attn(
#             x, x, x,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=False
#         )

#         x = self.norm1(x + x2)
#         x = self.norm2(x + self.ff(x))
#         return x



# class SAN(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         hidden_channels,
#         num_layers=4,
#         heads=4,
#         dropout=0.2,
#         pe_dim=20,
#         use_spd=True,
#         use_degree=True,
#         use_edge=False,
#     ):
#         super().__init__()

#         self.node_emb = nn.Linear(in_channels, hidden_channels)

#         self.pe_encoder = SignNet(pe_dim, hidden_channels)
#         self.pe_norm = nn.LayerNorm(hidden_channels)

#         self.struct_bias = SANStructuralBias(
#             pe_dim=pe_dim,
#             hidden_channels=hidden_channels,
#             heads=heads,
#             use_spd=use_spd,
#             use_degree=use_degree,
#             use_edge=use_edge,
#         )

#         self.layers = nn.ModuleList([
#             SANTransformerLayer(hidden_channels, heads, dropout)
#             for _ in range(num_layers)
#         ])

#         self.readout = global_mean_pool

#     def forward(self, x, edge_index, batch, pe, spd=None, degree=None, edge_attr=None):
#         x = self.node_emb(x)

#         pe_feat = self.pe_encoder(pe)
#         x = x + self.pe_norm(pe_feat)

#         x_dense, mask = to_dense_batch(x, batch)
#         pe_dense, _ = to_dense_batch(pe, batch)
#         spd_dense, _ = to_dense_batch(spd, batch) if spd is not None else (None, None)
#         deg_dense, _ = to_dense_batch(degree, batch) if degree is not None else (None, None)
#         edge_dense, _ = to_dense_batch(edge_attr, batch) if edge_attr is not None else (None, None)

#         key_padding_mask = ~mask

#         attn_bias = self.struct_bias(pe_dense, deg_dense, spd_dense, edge_dense)

#         for layer in self.layers:
#             x_dense = layer(x_dense, attn_bias, key_padding_mask)

#         x_sparse = x_dense[mask]
#         g = self.readout(x_sparse, batch)
#         return g, 0


# import torch
# from torch import nn
# from torch_geometric.utils import to_dense_batch, to_dense_adj
# from torch_geometric.nn import global_mean_pool


# class SignNet(nn.Module):
#     """
#     向量版 SignNet：
#     pe: [N, K] -> [N, H]
#     f(x) + f(-x) 保证符号不变性
#     """
#     def __init__(self, pe_dim: int, hidden_channels: int):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(pe_dim, hidden_channels),
#             nn.PReLU(),
#             nn.Linear(hidden_channels, hidden_channels),
#         )

#     def forward(self, pe: torch.Tensor) -> torch.Tensor:
#         if pe is None:
#             return None
#         out_pos = self.mlp(pe)      # [N, H]
#         out_neg = self.mlp(-pe)     # [N, H]
#         return out_pos + out_neg    # [N, H]


# class GraphormerBias(nn.Module):
#     """
#     Graphormer-style 结构偏置（head-wise scalar bias）：
#     - SPD bucket embedding
#     - degree (centrality) encoding
#     可选扩展 edge bias，但这里先保持简洁作为 SAN baseline。
#     """
#     def __init__(self, num_heads: int, spd_max_dist: int = 4, use_degree: bool = True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.spd_max_dist = spd_max_dist
#         self.use_degree = use_degree

#         # SPD bucket embedding: [0..spd_max_dist] -> R^{heads}
#         self.spd_emb = nn.Embedding(spd_max_dist + 1, num_heads)

#         # degree encoding: 度 -> R^{heads}
#         if use_degree:
#             self.degree_emb = nn.Embedding(512, num_heads)  # 假设度 < 512
#         else:
#             self.degree_emb = None

#     def forward(self, spd_dense: torch.Tensor, degree_dense: torch.Tensor = None) -> torch.Tensor:
#         """
#         spd_dense:    [B, N, N]  最短路径距离（整数或浮点，后者会被离散化）
#         degree_dense: [B, N]     节点度（整数）
#         return:       [B, heads, N, N]
#         """
#         B, N, _ = spd_dense.size()

#         # 1. SPD bucket
#         spd_bucket = torch.clamp(spd_dense.long(), max=self.spd_max_dist)  # [B, N, N]
#         spd_bias = self.spd_emb(spd_bucket)                                # [B, N, N, heads]

#         bias = spd_bias

#         # 2. degree bias（centrality）
#         if self.use_degree and degree_dense is not None:
#             deg_emb = self.degree_emb(degree_dense.long())  # [B, N, heads]
#             deg_i = deg_emb.unsqueeze(2)                    # [B, N, 1, heads]
#             deg_j = deg_emb.unsqueeze(1)                    # [B, 1, N, heads]
#             bias = bias + (deg_i + deg_j)                   # [B, N, N, heads]

#         # 3. reshape to [B, heads, N, N]
#         bias = bias.permute(0, 3, 1, 2).contiguous()
#         return bias


# class SANTransformerLayer(nn.Module):
#     def __init__(self, hidden_channels: int, heads: int, dropout: float):
#         super().__init__()
#         self.heads = heads

#         self.attn = nn.MultiheadAttention(
#             embed_dim=hidden_channels,
#             num_heads=heads,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.norm1 = nn.LayerNorm(hidden_channels)
#         self.ff = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_channels * 2, hidden_channels),
#             nn.Dropout(dropout),
#         )
#         self.norm2 = nn.LayerNorm(hidden_channels)

#     def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_padding_mask: torch.Tensor):
#         """
#         x:         [B, N, H]
#         attn_bias: [B, heads, N, N]  head-wise scalar bias
#         key_padding_mask: [B, N] (True = padding)
#         """
#         B, N, H = x.size()

#         # MultiheadAttention 的 attn_mask 需要 [B*heads, N, N]
#         attn_mask = attn_bias.reshape(B * self.heads, N, N)

#         x2, _ = self.attn(
#             x, x, x,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=False,
#         )

#         x = self.norm1(x + x2)
#         x = self.norm2(x + self.ff(x))
#         return x


# class SAN(nn.Module):
#     """
#     Graphormer-style SAN baseline：
#     - 节点特征 + SignNet(LPE)
#     - Graphormer-style 结构偏置（SPD + degree，head-wise scalar）
#     - Dense Transformer over nodes
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         num_layers: int = 4,
#         heads: int = 4,
#         dropout: float = 0.2,
#         pe_dim: int = 20,
#         spd_max_dist: int = 4,
#         use_degree: bool = True,
#     ):
#         super().__init__()

#         self.hidden_channels = hidden_channels
#         self.pe_dim = pe_dim

#         in_channels = max(in_channels, 1)

#         # 1. 节点特征编码
#         if in_channels != hidden_channels:
#             self.node_emb = nn.Linear(in_channels, hidden_channels)
#         else:
#             self.node_emb = nn.Identity()

#         # 2. LPE 编码（SignNet 向量版）
#         if pe_dim > 0:
#             self.pe_encoder = SignNet(pe_dim, hidden_channels)
#             self.pe_norm = nn.LayerNorm(hidden_channels)
#         else:
#             self.pe_encoder = None
#             self.pe_norm = None

#         # 3. Graphormer-style 结构偏置
#         self.struct_bias = GraphormerBias(
#             num_heads=heads,
#             spd_max_dist=spd_max_dist,
#             use_degree=use_degree,
#         )

#         # 4. Transformer 层
#         self.layers = nn.ModuleList([
#             SANTransformerLayer(hidden_channels, heads, dropout)
#             for _ in range(num_layers)
#         ])

#         # 5. 读出
#         self.readout = global_mean_pool

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

#     def forward(
#         self,
#         x: torch.Tensor,
#         edge_index,          # 保留接口，当前实现不显式使用
#         batch: torch.Tensor,
#         pe: torch.Tensor = None,       # [N, K] Laplacian eigenvectors
#         spd: torch.Tensor = None,      # [N, N] shortest path distance (precomputed)
#         degree: torch.Tensor = None,   # [N] node degree (precomputed)
#         *args,
#         **kwargs,
#     ):
#         """
#         x:      [N_total, F]
#         batch:  [N_total]
#         pe:     [N_total, K]
#         spd:    [N_total, N_total]（按图块拼接，需与 batch 对应）
#         degree: [N_total]
#         """

#         # 1. 节点特征编码
#         x = self.node_emb(x)  # [N, H]

#         # 2. LPE 注入
#         if self.pe_encoder is not None and pe is not None:
#             pe_feat = self.pe_encoder(pe)      # [N, H]
#             pe_feat = self.pe_norm(pe_feat)
#             x = x + pe_feat

#         # 3. Sparse -> Dense
#         x_dense, mask = to_dense_batch(x, batch)        # [B, N, H], [B, N]
#         key_padding_mask = ~mask                        # True = padding

#         # 4. 结构偏置输入：SPD + degree
#         if spd is None:
#             raise ValueError("Graphormer-style SAN 需要预先计算好的 spd 矩阵（[N, N]）")
#         # spd_dense, _ = to_dense_batch(spd, batch)       # [B, N, N]
#         spd_dense = to_dense_adj(edge_index, batch, spd)

#         if degree is not None:
#             deg_dense, _ = to_dense_batch(degree, batch)  # [B, N]
#         else:
#             deg_dense = None

#         # 5. Graphormer-style 结构偏置
#         attn_bias = self.struct_bias(spd_dense, deg_dense)  # [B, heads, N, N]

#         # 6. Transformer Stack
#         for layer in self.layers:
#             x_dense = layer(x_dense, attn_bias, key_padding_mask)

#         # 7. Dense -> Sparse & Readout
#         x_sparse = x_dense[mask]
#         g = self.readout(x_sparse, batch)

#         return g, 0
