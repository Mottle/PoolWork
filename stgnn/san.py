import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool


class SignNet(nn.Module):
    def __init__(self, pe_dim, hidden_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, pe):
        # pe: [N, K]
        if pe is None:
            return None
        # Vectorized implementation: f(x) + f(-x)
        return self.mlp(pe) + self.mlp(-pe)

# ------------------------------------------------------------
# 2. 修正后的 Bias 计算模块 (仅基于 LPE)
# ------------------------------------------------------------
class SANAttentionBias(nn.Module):
    """
    计算基于 LPE 差异的 Attention Bias
    """
    def __init__(self, hidden_channels, pe_dim, heads):
        super().__init__()
        self.heads = heads
        
        # LPE difference 编码
        self.lpe_mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, heads) # 直接输出 heads 个偏置
        )

    def forward(self, pe_dense):
        """
        pe_dense: [B, N_max, K]
        return:   [B*heads, N_max, N_max]  <-- 修正了形状
        """
        B, N, K = pe_dense.size()

        # 计算 Pair-wise LPE Difference
        # pe_i: [B, N, 1, K]
        # pe_j: [B, 1, N, K]
        pe_i = pe_dense.unsqueeze(2)
        pe_j = pe_dense.unsqueeze(1)
        
        # [B, N, N, K]
        lpe_diff = torch.abs(pe_i - pe_j)
        
        # [B, N, N, heads] -> [B, heads, N, N]
        bias = self.lpe_mlp(lpe_diff).permute(0, 3, 1, 2)
        
        # [B*heads, N, N] 符合 nn.MultiheadAttention 的要求
        return bias.reshape(B * self.heads, N, N)


class SANTransformerLayer(nn.Module):
    def __init__(self, hidden_channels, heads, dropout):
        super().__init__()
        self.heads = heads
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.ff = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(self, x, attn_bias, key_padding_mask):
        """
        x: [B, N, H]
        attn_bias: [B*heads, N, N]
        key_padding_mask: [B, N] (True where padding)
        """
        
        # [Fix] 将 Padding Mask 融合进 Attention Bias
        # key_padding_mask 是 [B, N]，我们需要把它扩展到 [B*heads, 1, N] 并加到 bias 上
        # PyTorch 的 MultiheadAttention 会自动处理 key_padding_mask，
        # 但既然我们自定义了 bias，最好确保 padding 位置是 -inf
        
        # 这里我们利用 nn.MultiheadAttention 自身的 key_padding_mask 参数即可
        # 只需要确保 attn_bias 形状正确
        
        x2, _ = self.attn(
            x, x, x,
            attn_mask=attn_bias, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + x2
        x = self.norm1(x)

        x2 = self.ff(x)
        x = x + x2
        x = self.norm2(x)

        return x


class SAN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=4,
        heads=4,
        dropout=0.2,
        pe_dim=20,
    ):
        super().__init__()
        
        # 必须保证 hidden_channels 能被 heads 整除
        assert hidden_channels % heads == 0

        self.node_emb = nn.Linear(in_channels, hidden_channels)
        
        # PE Encoder
        self.pe_encoder = SignNet(pe_dim, hidden_channels)
        self.pe_norm = nn.LayerNorm(hidden_channels)

        # 结构偏置模块 (Only LPE)
        self.struct_bias = SANAttentionBias(hidden_channels, pe_dim, heads)

        # Transformer 层
        self.layers = nn.ModuleList([
            SANTransformerLayer(hidden_channels, heads, dropout)
            for _ in range(num_layers)
        ])

        self.readout = global_mean_pool

    def forward(self, x, edge_index, batch, pe=None, *args, **kwargs):
        """
        x:   [N_total, F]
        pe:  [N_total, K]
        """
        # 1. 节点嵌入
        x = self.node_emb(x)

        # 2. LPE 注入 (Node Level)
        # SAN 原文既有 Node Level 的相加，又有 Attention Bias
        if pe is not None:
            pe_feat = self.pe_encoder(pe)
            pe_feat = self.pe_norm(pe_feat)
            x = x + pe_feat

        # 3. Sparse -> Dense
        # x_dense: [B, N_max, H]
        # mask:    [B, N_max] (True=Valid, False=Padding)
        x_dense, mask = to_dense_batch(x, batch)
        
        # pe_dense: [B, N_max, K]
        if pe is not None:
            pe_dense, _ = to_dense_batch(pe, batch)
        else:
            # 如果没有 PE，造一个全 0 的
            pe_dense = torch.zeros((x_dense.size(0), x_dense.size(1), 1), device=x.device)

        # key_padding_mask: True where padding (Transformer convention)
        key_padding_mask = ~mask 

        # 4. 计算结构偏置 (LPE Pair-wise Interaction)
        # 输出形状 [B*heads, N_max, N_max]
        if pe is not None:
            attn_bias = self.struct_bias(pe_dense)
        else:
            attn_bias = None

        # 5. Transformer Stack
        for layer in self.layers:
            x_dense = layer(x_dense, attn_bias, key_padding_mask)

        # 6. Dense -> Sparse
        x_sparse = x_dense[mask]

        # 7. Pooling & Classify
        g = self.readout(x_sparse, batch)
        
        return g, 0