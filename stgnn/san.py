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


import torch
from torch import nn
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import global_mean_pool


class SignNet(nn.Module):
    """
    向量版 SignNet：
    pe: [N, K] -> [N, H]
    f(x) + f(-x) 保证符号不变性
    """
    def __init__(self, pe_dim: int, hidden_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, pe: torch.Tensor) -> torch.Tensor:
        if pe is None:
            return None
        out_pos = self.mlp(pe)      # [N, H]
        out_neg = self.mlp(-pe)     # [N, H]
        return out_pos + out_neg    # [N, H]


class GraphormerBias(nn.Module):
    """
    Graphormer-style 结构偏置（head-wise scalar bias）：
    - SPD bucket embedding
    - degree (centrality) encoding
    可选扩展 edge bias，但这里先保持简洁作为 SAN baseline。
    """
    def __init__(self, num_heads: int, spd_max_dist: int = 4, use_degree: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.spd_max_dist = spd_max_dist
        self.use_degree = use_degree

        # SPD bucket embedding: [0..spd_max_dist] -> R^{heads}
        self.spd_emb = nn.Embedding(spd_max_dist + 1, num_heads)

        # degree encoding: 度 -> R^{heads}
        if use_degree:
            self.degree_emb = nn.Embedding(512, num_heads)  # 假设度 < 512
        else:
            self.degree_emb = None

    def forward(self, spd_dense: torch.Tensor, degree_dense: torch.Tensor = None) -> torch.Tensor:
        """
        spd_dense:    [B, N, N]  最短路径距离（整数或浮点，后者会被离散化）
        degree_dense: [B, N]     节点度（整数）
        return:       [B, heads, N, N]
        """
        B, N, _ = spd_dense.size()

        # 1. SPD bucket
        spd_bucket = torch.clamp(spd_dense.long(), max=self.spd_max_dist)  # [B, N, N]
        spd_bias = self.spd_emb(spd_bucket)                                # [B, N, N, heads]

        bias = spd_bias

        # 2. degree bias（centrality）
        if self.use_degree and degree_dense is not None:
            deg_emb = self.degree_emb(degree_dense.long())  # [B, N, heads]
            deg_i = deg_emb.unsqueeze(2)                    # [B, N, 1, heads]
            deg_j = deg_emb.unsqueeze(1)                    # [B, 1, N, heads]
            bias = bias + (deg_i + deg_j)                   # [B, N, N, heads]

        # 3. reshape to [B, heads, N, N]
        bias = bias.permute(0, 3, 1, 2).contiguous()
        return bias


class SANTransformerLayer(nn.Module):
    def __init__(self, hidden_channels: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.ff = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_padding_mask: torch.Tensor):
        """
        x:         [B, N, H]
        attn_bias: [B, heads, N, N]  head-wise scalar bias
        key_padding_mask: [B, N] (True = padding)
        """
        B, N, H = x.size()

        # MultiheadAttention 的 attn_mask 需要 [B*heads, N, N]
        attn_mask = attn_bias.reshape(B * self.heads, N, N)

        x2, _ = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = self.norm1(x + x2)
        x = self.norm2(x + self.ff(x))
        return x


class SAN(nn.Module):
    """
    Graphormer-style SAN baseline：
    - 节点特征 + SignNet(LPE)
    - Graphormer-style 结构偏置（SPD + degree，head-wise scalar）
    - Dense Transformer over nodes
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
        pe_dim: int = 20,
        spd_max_dist: int = 4,
        use_degree: bool = True,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.pe_dim = pe_dim

        in_channels = max(in_channels, 1)

        # 1. 节点特征编码
        if in_channels != hidden_channels:
            self.node_emb = nn.Linear(in_channels, hidden_channels)
        else:
            self.node_emb = nn.Identity()

        # 2. LPE 编码（SignNet 向量版）
        if pe_dim > 0:
            self.pe_encoder = SignNet(pe_dim, hidden_channels)
            self.pe_norm = nn.LayerNorm(hidden_channels)
        else:
            self.pe_encoder = None
            self.pe_norm = None

        # 3. Graphormer-style 结构偏置
        self.struct_bias = GraphormerBias(
            num_heads=heads,
            spd_max_dist=spd_max_dist,
            use_degree=use_degree,
        )

        # 4. Transformer 层
        self.layers = nn.ModuleList([
            SANTransformerLayer(hidden_channels, heads, dropout)
            for _ in range(num_layers)
        ])

        # 5. 读出
        self.readout = global_mean_pool

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index,          # 保留接口，当前实现不显式使用
        batch: torch.Tensor,
        pe: torch.Tensor = None,       # [N, K] Laplacian eigenvectors
        spd: torch.Tensor = None,      # [N, N] shortest path distance (precomputed)
        degree: torch.Tensor = None,   # [N] node degree (precomputed)
        *args,
        **kwargs,
    ):
        """
        x:      [N_total, F]
        batch:  [N_total]
        pe:     [N_total, K]
        spd:    [N_total, N_total]（按图块拼接，需与 batch 对应）
        degree: [N_total]
        """

        # 1. 节点特征编码
        x = self.node_emb(x)  # [N, H]

        # 2. LPE 注入
        if self.pe_encoder is not None and pe is not None:
            pe_feat = self.pe_encoder(pe)      # [N, H]
            pe_feat = self.pe_norm(pe_feat)
            x = x + pe_feat

        # 3. Sparse -> Dense
        x_dense, mask = to_dense_batch(x, batch)        # [B, N, H], [B, N]
        key_padding_mask = ~mask                        # True = padding

        # 4. 结构偏置输入：SPD + degree
        if spd is None:
            raise ValueError("Graphormer-style SAN 需要预先计算好的 spd 矩阵（[N, N]）")
        # spd_dense, _ = to_dense_batch(spd, batch)       # [B, N, N]
        spd_dense = to_dense_adj(edge_index, batch, spd)

        if degree is not None:
            deg_dense, _ = to_dense_batch(degree, batch)  # [B, N]
        else:
            deg_dense = None

        # 5. Graphormer-style 结构偏置
        attn_bias = self.struct_bias(spd_dense, deg_dense)  # [B, heads, N, N]

        # 6. Transformer Stack
        for layer in self.layers:
            x_dense = layer(x_dense, attn_bias, key_padding_mask)

        # 7. Dense -> Sparse & Readout
        x_sparse = x_dense[mask]
        g = self.readout(x_sparse, batch)

        return g, 0
