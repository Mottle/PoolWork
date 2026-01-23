import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import numpy as np
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch


class DeepSetSignNet(nn.Module):
    def __init__(self, pe_dim, hidden_channels):
        super().__init__()
        self.pe_dim = pe_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.post_proj = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, pe):
        if pe.size(1) > self.pe_dim:
            pe = pe[:, :self.pe_dim]
        pe = pe.unsqueeze(-1)  # [N, K, 1]
        
        # Sign Invariance
        out_pos = self.mlp(pe)
        out_neg = self.mlp(-pe)
        out = out_pos + out_neg
        
        # Sum Pooling + Post Proj
        out = out.sum(dim=1) 
        return self.post_proj(out)


class GraphGPS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1, # 新增：区分 Attention 的 dropout
        pe_dim: int = 20,
        use_edge_attr: bool = True, # GINE 必须需要 edge_attr
        heads: int = 4,
    ):
        super().__init__()

        self.in_channels = max(in_channels, 1)
        self.hidden_channels = hidden_channels
        self.use_edge_attr = use_edge_attr
        self.pe_dim = pe_dim

        # ---- Node Embedding ----
        if self.in_channels != hidden_channels:
            self.node_emb = nn.Linear(self.in_channels, hidden_channels)
        else:
            self.node_emb = nn.Identity()

        # ---- PE Encoder (保留你的 DeepSetSignNet) ----
        self.pe_encoder = DeepSetSignNet(pe_dim, hidden_channels) 

        # ---- Layers (替换为官方 GPSLayer) ----
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 实例化官方 GPSLayer
            layer = GPSLayer(
                dim_h=hidden_channels,
                local_gnn_type='GINE',         # <--- 指定 Backbone 为 GINE
                global_model_type='Transformer', 
                num_heads=heads,
                act='relu',                    # 官方默认常用 ReLU
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=False,              # 官方代码如果不改，通常二选一
                batch_norm=True,               # GraphGPS 常用 BatchNorm
                equivstable_pe=False           # 我们在外部处理 PE，这里设为 False
            )
            self.layers.append(layer)

        # ---- Readout ----
        self.readout = global_mean_pool
        
        # 官方代码通常在最后 Pooling 前不做额外的 Norm，
        # 但有些实现会加一个，这里保持简洁，与官方对齐。
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, batch, edge_attr=None, pe=None, *args, **kwargs):
        """
        输入输出签名保持不变。
        """
        # 1. 节点特征嵌入
        x = self.node_emb(x)

        # 2. PE 嵌入与叠加 (你的逻辑)
        if pe is not None:
            pe_proj = self.pe_encoder(pe)
            x = x + pe_proj # Add PE to features
        else:
            # 如果没有 PE，建议打印警告或报错，因为 GraphGPS 严重依赖 PE
            pass

        # -------------------------------------------------------------
        # 3. 构造伪 Batch 对象 (关键步骤)
        # 官方 GPSLayer 输入需要是 batch 对象，因为它要访问 batch.batch, batch.edge_index 等
        # -------------------------------------------------------------
        
        # 如果 Backbone 是 GINE，必须有 edge_attr。
        # 如果输入为 None，我们需要构造一个占位符或报错。
        if self.use_edge_attr and edge_attr is None:
            raise ValueError("GINE backbone requires edge_attr, but None provided.")
            
        # 构造临时的 PyG Batch 对象
        # 注意：这里的 x 已经是包含 PE 的 hidden_channels 维度向量
        batch_data = Batch(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            batch=batch
        )

        # 4. 通过官方 GPSLayers
        for layer in self.layers:
            # GPSLayer 输入输出都是 Batch 对象
            batch_data = layer(batch_data)

        # 5. 提取特征进行 Readout
        # 更新后的 x 存储在 batch_data.x 中
        final_x = batch_data.x
        
        g = self.readout(final_x, batch) 
        
        # 保持你的返回格式
        return g, 0



class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr,
                                                   batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'BiasedTransformer':
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h_attn = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
