import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix 
import scipy.sparse.linalg as sla
from torch_geometric.nn import GINEConv, GINConv
from torch_geometric.nn import global_mean_pool

class GraphGPSConv(nn.Module):
    r"""
    一个 GraphGPS block:
      x -> Local GNN -> +
                      Global Self-Attn -> Residual + Norm -> FFN -> Residual + Norm

    参数:
        channels:  节点特征维度 (输入 = 输出)
        local_gnn: 一个局部 GNN 模块, 例如 GINEConv/GATConv/GCNConv 等
        heads:     多头注意力头数
        attn_dropout: 自注意力内部 dropout
        dropout:   残差后的整体 dropout
        norm_type: 'layer' 或 'batch'
    """
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

        # 全局 Transformer-style self-attention (按 graph 作用)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=False,  # 我们用 (N, B, F) 形式
        )

        # FFN
        hidden = 2 * channels
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
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
        batch:      [num_nodes], 图索引
        edge_attr:  [num_edges, *] (可选, 传给 local_gnn)
        """
        # ---------- 局部 GNN ----------
        x_res = x
        if edge_attr is not None:
            x_local = self.local_gnn(x, edge_index, edge_attr)
        else:
            x_local = self.local_gnn(x, edge_index)

        # ---------- 全局 Self-Attention ----------
        # 将所有图打包成 (B, N_max, F) 再做 Transformer
        # mask: [B, N_max], True 表示有效节点
        x_dense, mask = to_dense_batch(x, batch)  # (B, N_max, F)
        B, N_max, F_dim = x_dense.size()

        # MultiheadAttention 输入为 (N_max, B, F)
        x_t = x_dense.transpose(0, 1)  # (N_max, B, F)

        # key_padding_mask: True 表示要 mask 的位置（即 padding）
        key_padding_mask = ~mask  # (B, N_max)

        x_attn, _ = self.self_attn(
            x_t, x_t, x_t,
            key_padding_mask=key_padding_mask
        )  # (N_max, B, F)

        x_attn = x_attn.transpose(0, 1)  # (B, N_max, F)

        # 去掉 padding，映射回 [num_nodes, F]
        x_global = x_attn[mask]  # (num_nodes, F)

        # ---------- 融合 + 残差 + Norm (Block 1: MPNN + Attn) ----------
        x = x_res + self.dropout(x_local + x_global)

        if self.norm_type == "batch":
            x = self.norm1(x)
        else:  # layer norm
            x = self.norm1(x)

        # ---------- FFN + 残差 + Norm (Block 2: FFN) ----------
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)

        if self.norm_type == "batch":
            x = self.norm2(x)
        else:
            x = self.norm2(x)

        return x



def build_gps_layer(channels: int, dropout: float = 0.1, edge_attr: bool = False):
    mlp = nn.Sequential(
        nn.Linear(channels, channels),
        nn.ReLU(),
        nn.Linear(channels, channels),
    )
    
    if edge_attr:
        local_gnn = GINEConv(mlp, edge_dim=channels)
    else:
        local_gnn = GINConv(mlp)

    gps_layer = GraphGPSConv(
        channels=channels,
        local_gnn=local_gnn,
        heads=4,
        attn_dropout=dropout,
        dropout=dropout,
        norm_type="layer",
    )
    return gps_layer



class LaplacianPE(nn.Module):
    def __init__(self, k: int = 20, normalization: str = "sym"):
        """
        k: 取前 k 个拉普拉斯特征向量
        normalization: "sym" / "rw" / None
        """
        super().__init__()
        self.k = k
        self.normalization = normalization

    def compute_lap_pe_single(self, edge_index, num_nodes):

        # 1. PyG 拉普拉斯
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, normalization=self.normalization, num_nodes=num_nodes
        )

        # 2. 转成 SciPy 稀疏矩阵
        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes=num_nodes)

        # 3. 取最小的 k 个特征向量
        k = min(self.k, num_nodes - 2)
        if k <= 0:
            return torch.zeros((num_nodes, self.k))

        eigvals, eigvecs = sla.eigsh(L, k=k, which="SM")
        pe = torch.from_numpy(eigvecs).float()  # CPU tensor

        # 如果节点数 < k，补零
        if pe.size(1) < self.k:
            pad = torch.zeros(num_nodes, self.k - pe.size(1))
            pe = torch.cat([pe, pad], dim=1)

        return pe

    def forward(self, x, edge_index, batch):
        device = x.device
        N = x.size(0)

        pe = torch.zeros((N, self.k), device=device)

        for g in batch.unique():
            mask = (batch == g)
            idx = mask.nonzero(as_tuple=False).view(-1)

            # 子图 edge_index
            sub_edge_mask = mask[edge_index[0]]
            sub_edge = edge_index[:, sub_edge_mask]

            new_index = torch.zeros(N, dtype=torch.long, device=device)
            new_index[idx] = torch.arange(idx.size(0), device=device)

            sub_edge = new_index[sub_edge]

            # 计算 LapPE（CPU）
            pe_g = self.compute_lap_pe_single(sub_edge.cpu(), idx.size(0))

            # 放回 GPU
            pe[idx] = pe_g.to(device)

        return pe

class SignNet(nn.Module):
    def __init__(self, pe_dim, hidden_channels):
        super().__init__()
        # 这里的 MLP 负责处理单个特征向量
        # 输入维度是 1 (每个 eigenvector 的一列)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 最后的线性层将所有特征向量的结果融合
        self.post_lin = nn.Linear(pe_dim * hidden_channels, hidden_channels)

    def forward(self, pe):
        """
        pe: [N, pe_dim] - 每个节点有 k 个特征向量分量
        """
        N, K = pe.shape
        outputs = []
        
        # 遍历每一个特征向量 (每一列)
        for i in range(K):
            v = pe[:, i:i+1] # [N, 1]
            
            # 符号不变性变换: f(v) + f(-v)
            out_pos = self.mlp(v)    # [N, hidden_channels]
            out_neg = self.mlp(-v)   # [N, hidden_channels]
            
            # 相加保证了无论输入 v 还是 -v，结果都一样
            out = out_pos + out_neg  # [N, hidden_channels]
            outputs.append(out)
            
        # 拼接所有特征向量的变换结果
        x_pe = torch.cat(outputs, dim=-1) # [N, K * hidden_channels]
        
        # 映射回模型所需的维度
        return self.post_lin(x_pe) # [N, hidden_channels]

class GraphGPS(nn.Module):
    """
    一个完整的 GraphGPS 模型：

      输入:
        - x:         [N, in_channels] 节点特征
        - edge_index:[2, E]
        - batch:     [N] 图索引
        - edge_attr: [E, edge_dim] (可选)
        - pe:        [N, pe_dim] (可选，预先计算好的 Laplacian PE)

      结构:
        x -> 节点嵌入 (Linear)
          -> (可选) LapPE 映射 + 相加
          -> L 层 GraphGPSConv 堆叠
          -> global_mean_pool
          -> MLP 分类头
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        pe_dim: int = 20,
        use_edge_attr: bool = False,
    ):
        super().__init__()

        self.in_channels = max(in_channels, 1)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe_dim = pe_dim
        self.use_edge_attr = use_edge_attr

        # ---- 节点嵌入 ----
        if self.in_channels != hidden_channels:
            self.node_emb = nn.Linear(self.in_channels, hidden_channels)
        else:
            self.node_emb = nn.Identity()

        self.pe_encoder = SignNet(pe_dim, hidden_channels) 
        self.pe_norm = nn.LayerNorm(hidden_channels)
        # self.edge_emb = nn.Linear(edge_in_dim, hidden_channels)

        # ---- 堆叠多个 GraphGPSConv 层 ----
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = build_gps_layer(
                channels=hidden_channels,
                dropout=dropout,
                edge_attr=use_edge_attr,
            )
            self.layers.append(layer)

        # ---- 读出 + 分类头 ----
        self.readout = global_mean_pool
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    # def push_pe(self, pe):
    #     self.pe = pe

    def forward(self, x, edge_index, batch, edge_attr=None, pe=None, *args, **kwargs):
        """
        x:         [N, in_channels]
        edge_index:[2, E]
        batch:     [N]
        edge_attr: [E, edge_dim] (可选, 仅在 use_edge_attr=True 时使用)
        """
        # ---- 节点嵌入 ----
        x = self.node_emb(x)

        pe_proj = self.pe_encoder(pe)
        pe_proj = self.pe_norm(pe_proj)
        x = x + pe_proj

        # ---- 多层 GraphGPSConv ----
        for layer in self.layers:
            if self.use_edge_attr:
                x = layer(x, edge_index, batch, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index, batch)

        # ---- 图级读出 ----
        g = self.readout(x, batch)  # [num_graphs, hidden_channels]

        # ---- 分类头 ----
        out = self.mlp_head(g)      # [num_graphs, out_channels]
        return out, 0
