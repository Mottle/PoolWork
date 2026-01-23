import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add
# import opt_einsum as oe
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from graph_gps import DeepSetSignNet

# =========================================================================
# 1. 基础辅助函数与定义
# =========================================================================

# 激活函数映射
act_dict = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'id': nn.Identity,
    'sigmoid': nn.Sigmoid,
}

def pyg_softmax(src, index, num_nodes=None):
    """稀疏 Softmax 实现"""
    num_nodes = maybe_num_nodes(index, num_nodes)
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out

@torch.no_grad()
def get_log_deg(batch):
    """计算节点的对数度数用于 Scaling"""
    if hasattr(batch, "log_deg"):
        return batch.log_deg.view(batch.num_nodes, 1)
    
    if hasattr(batch, "deg"):
        deg = batch.deg
    else:
        # 现场计算度数
        deg = pyg.utils.degree(batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.float)
        
    log_deg = torch.log(deg + 1).unsqueeze(-1)
    return log_deg.view(batch.num_nodes, 1)

# =========================================================================
# 3. GRIT 核心 Attention 模块
# =========================================================================

class MultiHeadAttentionLayerGritSparse(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, 
                 use_bias=True, clamp=5.0, dropout=0.0, act='relu',
                 edge_enhance=True):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        # Q, K, V, E 投影
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True) # 边特征维度 * 2
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        # 初始化
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        # Attention Weight 参数
        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        self.act = act_dict.get(act, nn.Identity)()

        # 边增强参数
        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]      # (E) x H x D
        dest = batch.Q_h[batch.edge_index[1]]     # (E) x H x D
        score = src + dest                        # Element-wise Sum

        # 融合边特征 (Key-Query Interaction with Edges)
        if hasattr(batch, "E") and batch.E is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
            
            # GRIT 核心公式: (Q+K) * E_w
            score = score * E_w
            # Signed Sqrt Activation
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b

        score = self.act(score)
        e_t = score # 保存用于更新边的中间状态

        # 输出更新后的边特征
        if hasattr(batch, "E") and batch.E is not None:
            batch.wE = score.flatten(1) # [E, H*D]

        # 计算 Attention Score
        # contraction: ehd, dhc -> ehc (D维度收缩)
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        # Softmax
        score = pyg_softmax(score, batch.edge_index[1], num_nodes=batch.num_nodes) 
        score = self.dropout(score)
        
        # 聚合 Value
        msg = batch.V_h[batch.edge_index[0]] * score  # (E) x H x D
        batch.wV = torch.zeros_like(batch.V_h)        # (N) x H x D
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        # 边特征增强聚合 (Edge Enhancement)
        if self.edge_enhance and hasattr(batch, "E") and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        V_h = self.V(batch.x)
        
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(batch)
        
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out

# =========================================================================
# 4. GRIT Transformer Layer
# =========================================================================

class GritTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True, act='relu',
                 norm_e=True, O_e=True,
                 # GRIT 特定配置 (原 cfg 内容)
                 update_e=True, deg_scaler=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.update_e = update_e
        self.deg_scaler = deg_scaler
        
        self.act = act_dict.get(act, nn.Identity)()

        # Attention 模块
        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=True,
            dropout=attn_dropout,
            clamp=5.0,
            act=act,
            edge_enhance=True
        )

        # Output Projection
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim) if O_e else nn.Identity()

        # Degree Scaler (GRIT 关键组件)
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

        # Normalization Layers (Block 1)
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim) if norm_e else nn.Identity()

        # FFN (Block 2)
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # Normalization Layers (Block 2)
        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h  
        e_in1 = batch.get("edge_attr", None)
        
        # 1. Self-Attention
        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # 2. Degree Scaler
        if self.deg_scaler:
            # stack: [N, D, 2] -> sum over last dim
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        
        e = None
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        # 3. Residual & Norm (Block 1)
        if self.residual:
            h = h_in1 + h
            if e is not None and e_in1 is not None:
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None: e = self.layer_norm1_e(e)
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None: e = self.batch_norm1_e(e)

        # 4. FFN (Block 2)
        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self.batch_norm2_h(h)

        # 5. 更新 Batch 状态
        batch.x = h
        if self.update_e and e is not None:
            batch.edge_attr = e
        else:
            # 如果不更新，保持原来的边特征
            batch.edge_attr = e_in1

        return batch

# =========================================================================
# 5. GRIT 主模型封装 (适配 NCI1 / TUDataset)
# =========================================================================

class GRIT(nn.Module):
    def __init__(
        self,
        in_channels: int,        # NCI1 = Atom Type 数量 (num_node_labels)
        hidden_channels: int,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.2,
        pe_dim: int = 20,
        num_edge_labels: int = 4 # NCI1 = Bond Type 数量
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # 1. Embedding
        # NCI1 节点是离散 label
        self.node_emb = nn.Embedding(in_channels, hidden_channels)
        # NCI1 边也是离散 label (GRIT 必须有 Edge Emb)
        self.edge_emb = nn.Embedding(num_edge_labels, hidden_channels)
        
        # 2. Positional Encoding
        self.pe_encoder = DeepSetSignNet(pe_dim, hidden_channels)

        # 3. Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GritTransformerLayer(
                    in_dim=hidden_channels,
                    out_dim=hidden_channels,
                    num_heads=heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    layer_norm=False,   # GRIT 默认不用 LayerNorm
                    batch_norm=True,    # 推荐用 BatchNorm
                    act='relu',
                    update_e=True,      # 开启边更新
                    deg_scaler=True     # 开启度数缩放
                )
            )

        # 4. Readout
        self.readout = global_mean_pool
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, batch, edge_attr=None, pe=None):
        # 1. Embeddings
        # 确保输入是 LongTensor (index)
        if x.dtype != torch.long: x = x.long()
        x = self.node_emb(x.squeeze())
        
        if edge_attr is None:
             # 如果没有边特征，造一个全0的
            edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        if edge_attr.dtype != torch.long: edge_attr = edge_attr.long()
        edge_attr = self.edge_emb(edge_attr.squeeze())

        # 2. Add PE
        if pe is not None:
            x = x + self.pe_encoder(pe)

        # 3. Construct Batch for GRIT Layers
        data = Batch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            num_nodes=x.size(0)
        )
        
        # 预计算 Log Degree (避免在 Layer 内部重复计算或报 Warning)
        deg = pyg.utils.degree(edge_index[1], num_nodes=x.size(0), dtype=torch.float)
        data.log_deg = torch.log(deg + 1).unsqueeze(-1)

        # 4. Layers Pass
        for layer in self.layers:
            data = layer(data)

        # 5. Readout
        g = self.readout(data.x, data.batch)
        
        # 返回 g 和 一个 dummy loss (保持和你之前接口一致)
        return g, 0