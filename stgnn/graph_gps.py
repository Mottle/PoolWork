import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GINEConv, GINConv
from torch_geometric.nn import global_mean_pool
from san import SignNet

# class SignNet(nn.Module):
#     """
#     [显存优化版] SignNet
#     将原本的 (Concat -> Linear) 改为 (MLP -> Sum)
#     利用 DeepSet 思想处理特征向量集合，大幅降低参数量和显存占用。
#     """
#     def __init__(self, pe_dim, hidden_channels):
#         super().__init__()
#         # 输入维度为 1 (标量)，处理每个特征向量的单个数值
#         self.mlp = nn.Sequential(
#             nn.Linear(1, hidden_channels),
#             nn.PReLU(),  # 使用 PReLU 或 LeakyReLU 均可
#             nn.Linear(hidden_channels, hidden_channels)
#         )
        
#     def forward(self, pe):
#         """
#         pe: [N, K] - N个节点，每个节点 K 个特征向量值
#         """
#         # 1. 升维以进行广播: [N, K] -> [N, K, 1]
#         pe = pe.unsqueeze(-1)
        
#         # 2. 向量化并行通过 MLP (替代循环)
#         # Linear 层作用于最后一个维度: [N, K, 1] -> [N, K, hidden_channels]
#         out_pos = self.mlp(pe)
#         out_neg = self.mlp(-pe)
        
#         # 3. 符号不变性处理 (f(x) + f(-x))
#         out = out_pos + out_neg  # [N, K, hidden_channels]
        
#         # 4. 聚合 (Sum Pooling): 将 K 个特征向量的信息融合
#         # [N, K, hidden_channels] -> [N, hidden_channels]
#         return out.sum(dim=1)

# class GraphGPSConv(nn.Module):
#     r"""
#     GraphGPS block:
#       x -> Local GNN -> +
#                       Global Self-Attn -> Residual + Norm -> FFN -> Residual + Norm
#     """
#     def __init__(
#         self,
#         channels: int,
#         local_gnn: MessagePassing,
#         heads: int = 4,
#         attn_dropout: float = 0.0,
#         dropout: float = 0.0,
#         norm_type: str = "layer",
#     ):
#         super().__init__()

#         self.channels = channels
#         self.local_gnn = local_gnn
#         self.dropout = nn.Dropout(dropout)

#         # [优化] 使用 batch_first=True，方便处理 to_dense_batch 的输出
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=channels,
#             num_heads=heads,
#             dropout=attn_dropout,
#             batch_first=True, 
#         )

#         # FFN
#         hidden = 2 * channels
#         self.ffn = nn.Sequential(
#             nn.Linear(channels, hidden),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, channels),
#         )

#         # Norm
#         if norm_type == "layer":
#             self.norm1 = nn.LayerNorm(channels)
#             self.norm2 = nn.LayerNorm(channels)
#         elif norm_type == "batch":
#             self.norm1 = nn.BatchNorm1d(channels)
#             self.norm2 = nn.BatchNorm1d(channels)
#         else:
#             raise ValueError(f"Unsupported norm_type: {norm_type}")

#         self.norm_type = norm_type

#     def forward(self, x, edge_index, batch, edge_attr=None):
#         """
#         x:          [num_nodes, channels]
#         edge_index: [2, num_edges]
#         batch:      [num_nodes]
#         edge_attr:  [num_edges, *]
#         """
#         # ---------- 1. Local GNN ----------
#         x_res = x
#         if edge_attr is not None:
#             x_local = self.local_gnn(x, edge_index, edge_attr)
#         else:
#             x_local = self.local_gnn(x, edge_index)

#         # ---------- 2. Global Self-Attention ----------
#         # 转换成稠密 batch: [B, N_max, F], mask: [B, N_max] (True表示有效节点)
#         x_dense, mask = to_dense_batch(x, batch)
        
#         # MultiheadAttention 的 key_padding_mask 中 True 表示需要被忽略(Padding)的位置
#         # 所以我们需要对 mask 取反
#         key_padding_mask = ~mask

#         # [显存优化] need_weights=False 不保存巨大的 Attention Map
#         x_attn, _ = self.self_attn(
#             x_dense, x_dense, x_dense,
#             key_padding_mask=key_padding_mask,
#             need_weights=False 
#         )

#         # 将有效节点提取回稀疏形式 [num_nodes, F]
#         x_global = x_attn[mask]

#         # ---------- 3. Fusion + Residual + Norm (Block 1) ----------
#         x = x_res + self.dropout(x_local + x_global)

#         if self.norm_type == "batch":
#             x = self.norm1(x)
#         else:
#             x = self.norm1(x)

#         # ---------- 4. FFN + Residual + Norm (Block 2) ----------
#         x_ffn = self.ffn(x)
#         x = x + self.dropout(x_ffn)

#         if self.norm_type == "batch":
#             x = self.norm2(x)
#         else:
#             x = self.norm2(x)

#         return x

# def build_gps_layer(channels: int, dropout: float = 0.1, edge_attr: bool = False):
#     # Local GNN 内部的 MLP
#     mlp = nn.Sequential(
#         nn.Linear(channels, channels),
#         nn.LeakyReLU(),
#         nn.Linear(channels, channels),
#     )
    
#     # 根据是否使用 edge_attr 选择 GINE 或 GIN
#     if edge_attr:
#         local_gnn = GINEConv(mlp, edge_dim=channels)
#     else:
#         local_gnn = GINConv(mlp)

#     gps_layer = GraphGPSConv(
#         channels=channels,
#         local_gnn=local_gnn,
#         heads=4,                 # 保持 4 头
#         attn_dropout=dropout,
#         dropout=dropout,
#         norm_type="layer",       # GT 通常首选 LayerNorm
#     )
#     return gps_layer

# class GraphGPS(nn.Module):
#     """
#     完整的 GraphGPS 模型
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         out_channels: int,
#         num_layers: int = 4,      # 默认改为 4 层，避免过深
#         dropout: float = 0.1,
#         pe_dim: int = 20,         # PE 维度
#         use_edge_attr: bool = False,
#     ):
#         super().__init__()

#         self.in_channels = max(in_channels, 1)
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.pe_dim = pe_dim
#         self.use_edge_attr = use_edge_attr

#         # ---- Node Embedding ----
#         # 将原始特征映射到 hidden_channels
#         if self.in_channels != hidden_channels:
#             self.node_emb = nn.Linear(self.in_channels, hidden_channels)
#         else:
#             self.node_emb = nn.Identity()

#         # ---- PE Encoder (SignNet) ----
#         self.pe_encoder = SignNet(pe_dim, hidden_channels) 
#         self.pe_norm = nn.LayerNorm(hidden_channels)

#         # ---- Layers ----
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             layer = build_gps_layer(
#                 channels=hidden_channels,
#                 dropout=dropout,
#                 edge_attr=use_edge_attr,
#             )
#             self.layers.append(layer)

#         # ---- Readout ----
#         self.readout = global_mean_pool
        
#         # ---- Classifier Head ----
#         # 通常 GT 后面接个 MLP 分类效果更好
#         self.mlp_head = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_channels // 2, out_channels),
#         )
        
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

#     def forward(self, x, edge_index, batch, edge_attr=None, pe=None, *args, **kwargs):
#         """
#         x:          [N, in_channels]
#         edge_index: [2, E]
#         batch:      [N]
#         edge_attr:  [E, edge_dim] (可选)
#         pe:         [N, pe_dim] (必须提供，由 DataLoader 计算)
#         """
#         # 1. 节点特征嵌入
#         x = self.node_emb(x)

#         # 2. PE 嵌入与叠加
#         if pe is not None:
#             # SignNet 处理
#             pe_proj = self.pe_encoder(pe)
#             pe_proj = self.pe_norm(pe_proj)
#             # 将 PE 加到节点特征上 (也可以选择 Concat，但 Add 更常见)
#             x = x + pe_proj
#         else:
#             print("Warning: No positional encodings (pe) provided to GraphGPS.")
#             pass

#         # 3. GraphGPS Layers
#         for layer in self.layers:
#             if self.use_edge_attr:
#                 x = layer(x, edge_index, batch, edge_attr=edge_attr)
#             else:
#                 x = layer(x, edge_index, batch)

#         # 4. Pooling
#         g = self.readout(x, batch)  # [batch_size, hidden_channels]

#         # 5. Classification
#         out = self.mlp_head(g)      # [batch_size, out_channels]
        
#         # 返回 logits 和 一个额外的 loss占位符 (兼容你之前的训练代码接口)
#         return out, 0



class GraphGPSConv(nn.Module):
    def __init__(
        self,
        channels: int,
        local_gnn: MessagePassing,
        heads: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        norm_type: str = "layer",
    ):
        super().__init__()

        self.channels = channels
        self.local_gnn = local_gnn
        self.dropout = nn.Dropout(dropout)

        # [显存优化] 使用 batch_first=True
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True, 
        )

        # FFN
        hidden = 2 * channels
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )

        # Norm
        if norm_type == "layer":
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
        elif norm_type == "batch":
            self.norm1 = nn.BatchNorm1d(channels)
            self.norm2 = nn.BatchNorm1d(channels)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.norm_type = norm_type

    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        x:          [num_nodes, channels]
        edge_index: [2, num_edges]
        batch:      [num_nodes]
        edge_attr:  [num_edges, *]
        """
        # ---------- 1. Local GNN ----------
        x_res = x
        if edge_attr is not None:
            x_local = self.local_gnn(x, edge_index, edge_attr)
        else:
            x_local = self.local_gnn(x, edge_index)

        # ---------- 2. Global Self-Attention ----------
        # Sparse -> Dense: [B, N_max, F]
        x_dense, mask = to_dense_batch(x, batch)
        
        # mask 中 True 是有效节点，MultiheadAttention 中 True 是 Padding
        key_padding_mask = ~mask

        # [显存优化] need_weights=False 不保存巨大的 Attention Map
        x_attn, _ = self.self_attn(
            x_dense, x_dense, x_dense,
            key_padding_mask=key_padding_mask,
            need_weights=False 
        )

        # Dense -> Sparse: [num_nodes, F]
        x_global = x_attn[mask]

        # ---------- 3. Fusion + Residual + Norm (Block 1) ----------
        x = x_res + self.dropout(x_local + x_global)

        if self.norm_type == "batch":
            x = self.norm1(x)
        else:
            x = self.norm1(x)

        # ---------- 4. FFN + Residual + Norm (Block 2) ----------
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)

        if self.norm_type == "batch":
            x = self.norm2(x)
        else:
            x = self.norm2(x)

        return x

def build_gps_layer(channels: int, dropout: float = 0.1, edge_attr: bool = False):
    # Local GNN 内部的 MLP
    mlp = nn.Sequential(
        nn.Linear(channels, channels),
        nn.LeakyReLU(),
        nn.Linear(channels, channels),
    )
    
    # 根据是否使用 edge_attr 选择 GINE 或 GIN
    if edge_attr:
        local_gnn = GINEConv(mlp, edge_dim=channels)
    else:
        local_gnn = GINConv(mlp)

    gps_layer = GraphGPSConv(
        channels=channels,
        local_gnn=local_gnn,
        heads=4,                 # 保持 4 头
        attn_dropout=dropout,
        dropout=dropout,
        norm_type="layer",       # GT 通常首选 LayerNorm
    )
    return gps_layer

# -------------------------------------------------------------------
# 3. GraphGPS Main Model
# -------------------------------------------------------------------
class GraphGPS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 4,      # 默认 4 层
        dropout: float = 0.1,
        pe_dim: int = 8,         # PE 维度 (K)
        use_edge_attr: bool = False,
    ):
        super().__init__()

        self.in_channels = max(in_channels, 1)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe_dim = pe_dim
        self.use_edge_attr = use_edge_attr

        # ---- Node Embedding ----
        if self.in_channels != hidden_channels:
            self.node_emb = nn.Linear(self.in_channels, hidden_channels)
        else:
            self.node_emb = nn.Identity()

        # ---- PE Encoder (集成 MLP-K 版 SignNet) ----
        # 注意：这里输入维度是 pe_dim (K)，输出维度是 hidden_channels (H)
        self.pe_encoder = SignNet(pe_dim, hidden_channels) 
        self.pe_norm = nn.LayerNorm(hidden_channels)

        # ---- Layers ----
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = build_gps_layer(
                channels=hidden_channels,
                dropout=dropout,
                edge_attr=use_edge_attr,
            )
            self.layers.append(layer)

        # ---- Readout ----
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

    def forward(self, x, edge_index, batch, edge_attr=None, pe=None, *args, **kwargs):
        """
        x:          [N, in_channels]
        pe:         [N, pe_dim] (必须提供，且 pe_dim 需与 init 中一致)
        """
        # 1. 节点特征嵌入
        x = self.node_emb(x)

        # 2. PE 嵌入与叠加
        if pe is not None:
            # SignNet (MLP-K) 处理: [N, K] -> [N, H]
            pe_proj = self.pe_encoder(pe)
            pe_proj = self.pe_norm(pe_proj)
            # 将 PE 加到节点特征上
            x = x + pe_proj
        else:
            # 如果没有 PE，建议打印警告
            # print("Warning: No positional encodings (pe) provided to GraphGPS.")
            pass

        # 3. GraphGPS Layers
        for layer in self.layers:
            if self.use_edge_attr:
                x = layer(x, edge_index, batch, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index, batch)

        # 4. Pooling
        g = self.readout(x, batch)  # [batch_size, hidden_channels]
        
        # 返回 logits 和 0 (辅助 loss 占位符)
        return g, 0