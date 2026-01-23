# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GINConv, global_mean_pool
# from torch_geometric.utils import to_dense_batch, to_dense_adj


# class StructureExtractor(nn.Module):
#     """
#     Structure Extractor module as described in the SAT paper.
#     Using GIN as the backbone to extract local structural information.
#     """

#     def __init__(self, hidden_dim, num_layers=3):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         for _ in range(num_layers):
#             mlp = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim * 2),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim * 2, hidden_dim),
#             )
#             self.convs.append(GINConv(mlp, train_eps=True))

#     def forward(self, x, edge_index, batch):
#         # Extract structure features
#         struct_feats = []
#         h = x
#         for conv in self.convs:
#             h = conv(h, edge_index)
#             h = F.relu(h)
#             struct_feats.append(h)

#         # In SAT, the structure representation is often the output of the GNN
#         # We take the final layer output as the structure representation H_str
#         return h


# class StructureAwareAttention(nn.Module):
#     """
#     Implements the Structure-Aware Self-Attention mechanism.
#     Attention(Q, K, V) = Softmax((QK^T / sqrt(d)) + StructureBias) V
#     """

#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         assert d_model % num_heads == 0

#         self.d_model = d_model
#         self.d_k = d_model // num_heads
#         self.num_heads = num_heads

#         # Projections for Content
#         self.W_Q = nn.Linear(d_model, d_model)
#         self.W_K = nn.Linear(d_model, d_model)
#         self.W_V = nn.Linear(d_model, d_model)

#         # Projections for Structure (generating the bias)
#         # We project structure features to heads to create structure-aware bias
#         self.W_S_Q = nn.Linear(d_model, d_model)
#         self.W_S_K = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)
#         self.out_proj = nn.Linear(d_model, d_model)

#     def forward(self, x_content, x_struct, mask=None):
#         """
#         x_content: [Batch, N, D] - Content features (Transformer flow)
#         x_struct:  [Batch, N, D] - Structure features (from GNN)
#         mask:      [Batch, N]    - Padding mask
#         """
#         B, N, _ = x_content.size()

#         # 1. Calculate Content Attention Scores
#         Q = (
#             self.W_Q(x_content).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
#         )  # [B, H, N, d_k]
#         K = self.W_K(x_content).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.W_V(x_content).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

#         content_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
#             self.d_k**0.5
#         )  # [B, H, N, N]

#         # 2. Calculate Structure Bias
#         # The paper suggests measuring similarity between structural representations
#         Q_s = self.W_S_Q(x_struct).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
#         K_s = self.W_S_K(x_struct).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

#         # This term acts as psi(u, v) in the paper
#         structure_bias = torch.matmul(Q_s, K_s.transpose(-2, -1)) / (
#             self.d_k**0.5
#         )  # [B, H, N, N]

#         # 3. Combine and Softmax
#         attn_scores = content_scores + structure_bias

#         if mask is not None:
#             # mask is [B, N], expand to [B, 1, 1, N] for broadcasting
#             # We want to mask out positions where mask is False (padding)
#             mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
#             attn_scores = attn_scores.masked_fill(~mask_expanded, float("-inf"))

#         attn_probs = F.softmax(attn_scores, dim=-1)
#         attn_probs = self.dropout(attn_probs)

#         output = torch.matmul(attn_probs, V)  # [B, H, N, d_k]
#         output = output.transpose(1, 2).contiguous().view(B, N, self.d_model)

#         return self.out_proj(output)


# class SATLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         self.attention = StructureAwareAttention(d_model, num_heads, dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model),
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, x_struct, mask=None):
#         # Pre-Norm or Post-Norm (Standard Transformer is usually Post-Norm, but Pre-Norm is stable for Deep GNNs)
#         # Here we use Post-Norm standard implementation
#         attn_out = self.attention(x, x_struct, mask)
#         x = self.norm1(x + self.dropout(attn_out))

#         ffn_out = self.ffn(x)
#         x = self.norm2(x + self.dropout(ffn_out))
#         return x


# class SAT(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         hidden_channels=64,
#         num_layers=3,
#         num_heads=4,
#         rw_steps=20,
#         dropout=0.1,
#     ):
#         super().__init__()

#         # 1. Embeddings
#         self.node_emb = nn.Linear(in_channels, hidden_channels)

#         # Random Walk Positional Encoding (Crucial for SAT)
#         # Assuming input includes pre-computed RWPE features or we project raw RWPE if provided
#         # For this baseline, we assume raw features are concatenated with RWPE or processed separately.
#         # Here we implement a simple learnable embedding for RWPE if provided in x,
#         # otherwise we assume x already contains PE.
#         # To be safe and explicit:
#         self.rwpe_proj = nn.Linear(rw_steps, hidden_channels)

#         # 2. Structure Extractor (GIN)
#         self.structure_extractor = StructureExtractor(hidden_channels, num_layers=3)

#         # 3. Transformer Layers
#         self.layers = nn.ModuleList(
#             [SATLayer(hidden_channels, num_heads, dropout) for _ in range(num_layers)]
#         )

#     def forward(self, x, edge_index, batch, rw_pos_enc=None, *args, **kwargs):
#         """
#         x: [Num_Nodes, Feature_Dim]
#         edge_index: [2, Num_Edges]
#         batch: [Num_Nodes]
#         rw_pos_enc: [Num_Nodes, RW_Steps] (Random Walk PE stats)
#         """

#         # Initial Embedding
#         h = self.node_emb(x)

#         # Add Positional Encoding (Paper Requirement)
#         if rw_pos_enc is not None:
#             pe = self.rwpe_proj(rw_pos_enc)
#             h = h + pe

#         # Extract Structure Features (The "k-subtree" representation)
#         # We pass the full graph to the GNN extractor
#         h_struct = self.structure_extractor(h, edge_index, batch)

#         # Convert to Dense Batch for Transformer
#         # Transformer calculates attention between ALL nodes in a graph
#         # [Batch_Size, Max_Nodes, Hidden_Dim]
#         h_dense, mask = to_dense_batch(h, batch)
#         h_struct_dense, _ = to_dense_batch(h_struct, batch)

#         # Transformer Pass
#         for layer in self.layers:
#             # We pass both content (h_dense) and structure (h_struct_dense)
#             h_dense = layer(h_dense, h_struct_dense, mask=mask)

#         # Mask out padding nodes before pooling to avoid zeros affecting mean
#         # h_dense is [B, N_max, D], mask is [B, N_max]
#         # We need to reconstruct the flat batch for global_mean_pool or pool manually

#         # Method A: Convert back to sparse for standard PyG pooling (Safest)
#         h_flat = h_dense[mask]  # Select valid nodes
#         batch_reconstructed = batch  # Original batch vector is aligned with x

#         # Global Mean Pooling (Direct Output as requested)
#         graph_emb = global_mean_pool(h_flat, batch_reconstructed)

#         return graph_emb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

class StructureExtractor(nn.Module):
    # ... (保持不变，这部分显存占用很小)
    def __init__(self, hidden_dim, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
        
    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return h

class OptimizedSATAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Content Projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # Structure Projections
        self.W_S_Q = nn.Linear(d_model, d_model)
        self.W_S_K = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x_content, x_struct, mask=None):
        B, N, _ = x_content.size()
        
        # 1. 投影
        # [B, N, H, D_k]
        q_c = self.W_Q(x_content).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k_c = self.W_K(x_content).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v_c = self.W_V(x_content).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        q_s = self.W_S_Q(x_struct).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k_s = self.W_S_K(x_struct).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 优化核心：拼接 (Concatenation) 代替 加法
        # 数学等价转换：Qc*Kc + Qs*Ks = [Qc, Qs] * [Kc, Ks]
        # 我们在特征维度拼接：新的 d_k 变为 2 * d_k
        q_all = torch.cat([q_c, q_s], dim=-1) # [B, H, N, 2*d_k]
        k_all = torch.cat([k_c, k_s], dim=-1) # [B, H, N, 2*d_k]
        
        # 3. 缩放修正 (Scale Correction)
        # SAT 原文公式除以 sqrt(d_k)。
        # F.scaled_dot_product_attention 默认除以 sqrt(dim)，这里 dim 是 2*d_k。
        # 为了保持数学一致性：我们需要手动把输入放大 sqrt(2)，这样 sqrt(2)/sqrt(2*dk) = 1/sqrt(dk)
        q_all = q_all * (2.0 ** 0.5) 

        # 4. 使用 PyTorch 2.0+ 的 FlashAttention
        # 这会自动处理显存优化，避免生成 [B, H, N, N] 的中间矩阵
        if mask is not None:
            # mask 是 [B, N]，True表示有效节点。
            # SDPA 需要 mask 形状广播匹配。
            # 注意：SDPA 的 attn_mask 语义：True 的位置保留，False 的位置变为 -inf (如果是 Bool)
            # 或者 0 处加 0， -inf 处加 -inf (如果是 Float)
            # PyTorch SDPA 推荐使用 attn_mask=None 并用 is_causal=False 来获得最大加速，
            # 但对于 Padding Mask，我们必须传。
            
            # [B, 1, 1, N] -> 扩展到 [B, H, N, N] 的广播
            attn_mask = mask.view(B, 1, 1, N).expand(B, self.num_heads, N, N)
            
            # 使用 PyTorch 优化的 SDPA
            output = F.scaled_dot_product_attention(
                q_all, k_all, v_c, 
                attn_mask=attn_mask, 
                dropout_p=self.dropout_p if self.training else 0.0
            )
        else:
             output = F.scaled_dot_product_attention(
                q_all, k_all, v_c, 
                dropout_p=self.dropout_p if self.training else 0.0
            )

        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(output)

class SATLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = OptimizedSATAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_struct, mask=None):
        # 这里的 x 和 x_struct 都是 Dense Batch [B, N, D]
        attn_out = self.attention(x, x_struct, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class SAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, num_heads=4, rw_steps=20, dropout=0.1, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing # 开关
        
        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.rwpe_proj = nn.Linear(rw_steps, hidden_channels)
        self.structure_extractor = StructureExtractor(hidden_channels, num_layers=3)
        
        self.layers = nn.ModuleList([
            SATLayer(hidden_channels, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_channels, hidden_channels) # 示例分类头

    def forward(self, x, edge_index, batch, rw_pos_enc=None, *args, **kwargs):
        h = self.node_emb(x)
        if rw_pos_enc is not None:
            h = h + self.rwpe_proj(rw_pos_enc)
            
        h_struct = self.structure_extractor(h, edge_index, batch)
        
        # 依然使用 Dense Batch，但后续计算优化了
        h_dense, mask = to_dense_batch(h, batch)
        h_struct_dense, _ = to_dense_batch(h_struct, batch)
        
        for layer in self.layers:
            if self.use_checkpointing:
                # 显存救星：以计算换显存
                # 只在反向传播时重算中间激活值，能节省约 50%-70% 的显存
                h_dense = torch.utils.checkpoint.checkpoint(
                    layer, h_dense, h_struct_dense, mask,
                    use_reentrant=False 
                )
            else:
                h_dense = layer(h_dense, h_struct_dense, mask)

        # 还原回 Sparse 做 Pooling (避免计算无效节点的 mean)
        h_flat = h_dense[mask]
        out = global_mean_pool(h_flat, batch)
        
        return out, 0