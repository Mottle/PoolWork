import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add
import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
# from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import remove_self_loops
from torch_geometric.graphgym.register import *
import opt_einsum as oe
from yacs.config import CfgNode as CN
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data

import warnings
# # https://github.com/LiamMa/GRIT/blob/6c988ea600a606fbb49a2246c64a2d37396b3ab5/grit/layer/grit_layer.py#L144


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
                        dim=0, dim_size=batch_size, reduce='add')
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
        scatter(zero, idx, dim=0, out=adj, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out



class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
        Proposed Attention Computation for GRIT
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=CN(),
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]      # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]     # (num relative) x num_heads x out_dim
        score = src + dest                        # element-wise multiplication

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
            # (num relative) x num_heads x out_dim
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        raw_attn = score
        score = pyg_softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h, dtype=msg.dtype)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.edge_enhance and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add", dim_size=batch.wV.size(0))
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


@register_layer("GritTransformer")
class GritTransformerLayer(nn.Module):
    """
        Proposed Transformer Layer for GRIT
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 cfg=dict(),
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # -------
        self.update_e = cfg.get("update_e", True)
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        # self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            act=cfg.attn.get("act", "relu"),
            edge_enhance=cfg.attn.get("edge_enhance", True),
            sqrt_relu=cfg.attn.get("sqrt_relu", False),
            signed_sqrt=cfg.attn.get("signed_sqrt", False),
            scaled_attn =cfg.attn.get("scaled_attn", False),
            no_qk=cfg.attn.get("no_qk", False),
        )

        # if cfg.attn.get('graphormer_attn', False):
        #     self.attention = MultiHeadAttentionLayerGraphormerSparse(
        #         in_dim=in_dim,
        #         out_dim=out_dim // num_heads,
        #         num_heads=num_heads,
        #         use_bias=cfg.attn.get("use_bias", False),
        #         dropout=attn_dropout,
        #         clamp=cfg.attn.get("clamp", 5.),
        #         act=cfg.attn.get("act", "relu"),
        #         edge_enhance=True,
        #         sqrt_relu=cfg.attn.get("sqrt_relu", False),
        #         signed_sqrt=cfg.attn.get("signed_sqrt", False),
        #         scaled_attn =cfg.attn.get("scaled_attn", False),
        #         no_qk=cfg.attn.get("no_qk", False),
        #     )



        self.O_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        # -------- Deg Scaler Option ------

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1,1))
            self.alpha2_h = nn.Parameter(torch.zeros(1,1))
            self.alpha1_e = nn.Parameter(torch.zeros(1,1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None
        # multi-head attention out

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero: h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero: e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None: e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None: e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero: h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.edge_index[1],
                               num_nodes=batch.num_nodes,
                               dtype=torch.float
                               )
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


# @register_network('GritTransformer')
# class GritTransformer(torch.nn.Module):
#     '''
#         The proposed GritTransformer (Graph Inductive Bias Transformer)
#     '''

#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.encoder = FeatureEncoder(dim_in)
#         dim_in = self.encoder.dim_in

#         self.ablation = True
#         self.ablation = False

#         if cfg.posenc_RRWP.enable:
#             self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\
#                 (cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)
#             rel_pe_dim = cfg.posenc_RRWP.ksteps
#             self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \
#                 (rel_pe_dim, cfg.gnn.dim_edge,
#                  pad_to_full_graph=cfg.gt.attn.full_attn,
#                  add_node_attr_as_self_loop=False,
#                  fill_value=0.
#                  )


#         if cfg.gnn.layers_pre_mp > 0:
#             self.pre_mp = GNNPreMP(
#                 dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
#             dim_in = cfg.gnn.dim_inner

#         assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
#             "The inner and hidden dims must match."

#         global_model_type = cfg.gt.get('layer_type', "GritTransformer")
#         # global_model_type = "GritTransformer"

#         TransformerLayer = register.layer_dict.get(global_model_type)

#         layers = []
#         for l in range(cfg.gt.layers):
#             layers.append(TransformerLayer(
#                 in_dim=cfg.gt.dim_hidden,
#                 out_dim=cfg.gt.dim_hidden,
#                 num_heads=cfg.gt.n_heads,
#                 dropout=cfg.gt.dropout,
#                 act=cfg.gnn.act,
#                 attn_dropout=cfg.gt.attn_dropout,
#                 layer_norm=cfg.gt.layer_norm,
#                 batch_norm=cfg.gt.batch_norm,
#                 residual=True,
#                 norm_e=cfg.gt.attn.norm_e,
#                 O_e=cfg.gt.attn.O_e,
#                 cfg=cfg.gt,
#             ))
#         # layers = []

#         self.layers = torch.nn.Sequential(*layers)
#         GNNHead = register.head_dict[cfg.gnn.head]
#         self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

#     def forward(self, batch):
#         for module in self.children():
#             batch = module(batch)

#         return batch






class ConfigWrapper:
    """
    一个简单的辅助类，允许通过点号(.)访问字典属性，
    用于模拟 GraphGym 的 cfg 对象，适配 GritTransformerLayer 的接口。
    """
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                v = ConfigWrapper(v)
            self.__dict__[k] = v
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)
    
    def __getattr__(self, name):
        return self.__dict__.get(name, None)

class GRIT(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.2,
        rrwp_steps: int = 20,       # RRWP 的步数 (K)
        use_rrwp: bool = True,      # 是否启用 RRWP
        num_edge_labels: int = None, # 如果边是离散标签，传入类别数；如果是连续向量，设为 None
        edge_in_channels: int = 0,   # 如果边是连续向量，传入维度
        act: str = 'relu',
        norm_e: bool = True,
        O_e: bool = True,
        pool: str = 'mean'
    ):
        super().__init__()
        self.use_rrwp = use_rrwp
        self.hidden_channels = hidden_channels
        self.pooling_type = pool

        # ---------------------------------------------------------
        # 1. Embeddings (Node & Edge & RRWP)
        # ---------------------------------------------------------
        
        # Node Embedding
        # 假设输入 x 已经是特征向量。如果是离散索引，请改用 nn.Embedding
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # Edge Embedding
        if num_edge_labels is not None:
            self.edge_encoder = nn.Embedding(num_edge_labels, hidden_channels)
        elif edge_in_channels > 0:
            self.edge_encoder = nn.Linear(edge_in_channels, hidden_channels)
        else:
            self.edge_encoder = None # 无原始边特征
            
        # RRWP Encoders
        if self.use_rrwp:
            # 相对位置编码 (Relative PE): 处理边上的 RRWP (E, K)
            self.rrwp_rel_encoder = nn.Linear(rrwp_steps, hidden_channels)
            
            # 绝对位置编码 (Absolute PE): 处理节点上的 RRWP 对角线 (N, K)
            # 注意：如果你的 DataLoader 没有提供 rrwp_abs，这部分可以注释掉
            self.rrwp_abs_encoder = nn.Linear(rrwp_steps, hidden_channels)

        # ---------------------------------------------------------
        # 2. Config Wrapper for Layers
        # ---------------------------------------------------------
        # 构造一个配置对象传给 Layer，模拟 GraphGym 的 cfg
        layer_cfg_dict = {
            "bn_momentum": 0.1,
            "bn_no_runner": False,
            "rezero": False,
            "update_e": True,
            "attn": {
                "use": True,
                "deg_scaler": True,
                "use_bias": False,
                "clamp": 5.0,
                "act": act,
                "edge_enhance": True,
                "sqrt_relu": False,
                "signed_sqrt": False,
                "scaled_attn": False,
                "no_qk": False,
                "norm_e": norm_e,
                "O_e": O_e
            }
        }
        self.layer_cfg = ConfigWrapper(layer_cfg_dict)

        # ---------------------------------------------------------
        # 3. Transformer Layers
        # ---------------------------------------------------------
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GritTransformerLayer(
                    in_dim=hidden_channels,
                    out_dim=hidden_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    layer_norm=False,      # GRIT 默认配置
                    batch_norm=True,       # GRIT 默认配置
                    residual=True,
                    act=act,
                    norm_e=norm_e,
                    O_e=O_e,
                    cfg=self.layer_cfg     # 传入模拟的 cfg
                )
            )

        # ---------------------------------------------------------
        # 4. Pooling / Readout
        # ---------------------------------------------------------
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, batch, edge_attr=None, rrwp=None, rrwp_abs=None, *args, **kargs):
        """
        Args:
            x (Tensor): 节点特征 [N, In_Dim]
            edge_index (LongTensor): 边索引 [2, E]
            batch (LongTensor): 批次索引向量 [N]
            edge_attr (Tensor, optional): 原始边特征 [E, Edge_Dim]
            rrwp (Tensor, optional): 相对位置编码 (Relative RRWP) [E, K]
            rrwp_abs (Tensor, optional): 绝对位置编码 (Absolute RRWP/RWSE) [N, K]
        """
        
        # 1. Encode Node Features
        # 假设输入 x 已经是 float 特征。如果是离散索引，请确保初始化时用了 Embedding
        x = self.node_encoder(x)

        # 2. Encode Edge Features (Initial)
        # 初始化 edge_attr
        if edge_attr is None:
            # 如果没有提供边特征，创建一个全 0 向量作为初始槽位
            # GRIT 需要边特征槽位来通过 Attention 更新信息
            edge_attr = torch.zeros(
                (edge_index.size(1), self.hidden_channels), 
                device=x.device, dtype=x.dtype
            )
        else:
            # 如果提供了边特征，且定义了 edge_encoder，则进行编码
            if self.edge_encoder is not None:
                edge_attr = self.edge_encoder(edge_attr)
        
        # 3. Inject RRWP (关键步骤)
        if self.use_rrwp:
            # (A) 相对 RRWP -> 加到边特征 (E, K) -> (E, H)
            if rrwp is not None:
                rrwp_embed = self.rrwp_rel_encoder(rrwp) 
                edge_attr = edge_attr + rrwp_embed
            
            # (B) 绝对 RRWP -> 加到节点特征 (N, K) -> (N, H)
            if rrwp_abs is not None:
                rrwp_abs_embed = self.rrwp_abs_encoder(rrwp_abs) 
                x = x + rrwp_abs_embed

        # 4. Construct Data/Batch Object for Layers
        # 由于 GritTransformerLayer 内部使用 batch.x, batch.edge_index 等方式访问
        # 我们需要将处理好的 Tensor 封装回一个 PyG Data 对象中
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            batch=batch, 
            num_nodes=x.size(0)
        )

        # 5. Degree Scaler Preparation
        # 确保 data 中有 log_deg (Degree Scaler 需要)
        # get_log_deg 是之前定义的辅助函数，它接受一个对象并读取 edge_index 和 num_nodes
        if not hasattr(data, 'log_deg'):
             data.log_deg = get_log_deg(data) 

        # 6. Pass through Layers
        for layer in self.layers:
            data = layer(data)

        # 7. Readout (Pooling)
        # 从 data 对象中取出最终的节点特征 x 和批次索引 batch 进行聚合
        h_graph = self.pool(data.x, data.batch)

        return h_graph, 0