import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, GraphNorm, knn_graph
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_geometric.data import Data
import networkx as nx
from perf_counter import get_time_sync
from torch import Tensor
from typing import Optional
from torch_geometric.utils import scatter


class DualRoadGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5, k = 3, backbone = 'gcn'):
        super(DualRoadGNN, self).__init__()
        self.in_channels = max(in_channels, 1)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k
        self.backbone = backbone

        if num_layers < 2:
            raise ValueError("Number of layers should be greater than 1.")
        
        if k <= 1:
            raise ValueError("k should be greater than 1.")
        
        self._build_embedding()
        self.convs = self._build_convs()
        self.norms = self._build_graph_norms()
        self.feature_convs = self._build_convs()
        self.feature_norms = self._build_graph_norms()
        self.fusion_gate_linear = nn.Linear(self.hidden_channels * 2, hidden_channels)
        # self.fusion_dense_mlp = nn.Sequential(
        #     nn.Linear(self.hidden_channels * 2, self.hidden_channels * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(2 * self.hidden_channels, self.hidden_channels),
        #     nn.Dropout(p=self.dropout)
        # )
        # self.fusion_gate_mlp = nn.Sequential(
        #     nn.Linear(self.hidden_channels * 2, self.hidden_channels * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(2 * self.hidden_channels, self.hidden_channels),
        #     nn.Dropout(p=self.dropout)
        # )
    
    def _build_embedding(self):
        # self.embedding = nn.Embedding(num_embeddings=self.in_channels, embedding_dim=self.hidden_channels)
        self.embedding = nn.Linear(in_features=self.in_channels, out_features=self.hidden_channels)

    def _build_convs(self):
        convs = nn.ModuleList()
        for i in range(self.num_layers):
            if self.backbone == 'gcn':
                convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
            elif self.backbone == 'gin':
                fnn = nn.Linear(self.hidden_channels, self.hidden_channels)
                convs.append(GINConv(fnn))
            elif self.backbone == 'gat':
                convs.append(GATConv(self.hidden_channels, self.hidden_channels // 4, heads=4))
        return convs
    
    def _build_feature_convs(self):
        convs = nn.ModuleList()
        for i in range(self.num_layers):
            if self.backbone == 'gcn':
                convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
            elif self.backbone == 'gin':
                fnn = nn.Linear(self.hidden_channels, self.hidden_channels)
                convs.append(GINConv(fnn))
            elif self.backbone == 'gat':
                convs.append(GATConv(self.hidden_channels, self.hidden_channels // 4, heads=4))
        return convs

    def _build_graph_norms(self):
        graph_norms = nn.ModuleList()
        for i in range(self.num_layers):
            graph_norms.append(GraphNorm(self.hidden_channels))
        return graph_norms
    
    def _build_auxiliary_graph(self, x, batch):
        feature_graph_edge_index = knn_graph(x, self.k, batch, loop=True, cosine=True)
        return feature_graph_edge_index

    def forward(self, x, edge_index, batch):
        originl_x = x
        x = self.embedding(x)
        
        feature_graph_edge_index = self._build_auxiliary_graph(x, batch)

        all_x = []  

        for i in range(self.num_layers - 1):
            prev_x = x

            x = self.convs[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            feature_x = self.feature_convs[i](x, feature_graph_edge_index)
            feature_x = self.feature_norms[i](feature_x, batch)
            feature_x = F.leaky_relu(feature_x)
            feature_x = F.dropout(feature_x, p=self.dropout, training=self.training)

            combined = torch.cat([x, feature_x], dim=-1)
            gate = torch.sigmoid(self.fusion_gate_linear(combined))

            fusion_x = gate * x + (1 - gate) * feature_x + prev_x

            # fusion_x = self.fusion_dense_mlp(combined) + prev_x
            all_x.append(fusion_x)
            x = fusion_x

        graph_feature = 0
        for i in range(self.num_layers):
            graph_feature += global_mean_pool(all_x[i - 1], batch)
        return graph_feature, 0
    

class KFNDualRoadGNN(DualRoadGNN):
    def _build_auxiliary_graph(self, x, batch):
        return k_farthest_graph(x, self.k, batch, loop=True, cosine=True, direction=True)
    

class KFNDualRoadSTSplitGNN(DualRoadGNN):
    def _build_auxiliary_graph(self, x, batch):
        return k_farthest_graph(x, self.k, batch, loop=True, cosine=True, direction=True)
    
    def _build_convs(self):
        from st_split_gnn import SpanTreeSplitGNN, SpanTreeSplitConv
        convs = nn.ModuleList()
        for i in range(self.num_layers):
            # convs.append(SpanTreeSplitGNN(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=1, num_splits=2, dropout=self.dropout))
            convs.append(SpanTreeSplitConv(in_channels=self.hidden_channels, out_channels=self.hidden_channels, num_splits=2, dropout=self.dropout, backbone=self.backbone))
        return convs


class KRNDualRoadGNN(DualRoadGNN):
    def _build_auxiliary_graph(self, x, batch):
        return k_random_graph(x, self.k, batch, loop=True)

def k_farthest_graph(
    x: Tensor,
    k: int,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    cosine: bool = True,
    direction: bool = True, #默认单向
) -> Tensor:
    """
    计算基于特征空间中 K-最远邻居的图的边索引 (edge_index)，支持批处理。

    参数:
        x (torch.Tensor): 节点特征矩阵，形状为 [num_nodes, num_features]。
        k (int): 要连接的最远邻居的数量。
        batch (torch.Tensor, optional): 批次向量，将节点映射到对应的图。
                                        如果为 None，则假设所有节点属于单个图。
        loop (bool, optional): 如果为 True，则图中包含自环 (self-loops)。
        cosine (bool, optional): 如果为 True，则使用余弦距离；否则使用欧几里得距离。

    返回:
        torch.Tensor: 图的边索引 (edge_index)，形状为 [2, num_edges]。
    """
    # 如果 batch 为 None，我们创建一个单图 batch 向量，并调用核心逻辑
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    
    num_nodes = x.size(0)
    
    # 获取每个图的节点数 (n_i)
    num_graphs = batch.max().item() + 1
    
    # PyTorch Geometric 的 'ptr' (Pointer) 机制，用于定位批次中的子图
    # [0, n1, n1+n2, n1+n2+n3, ..., N]
    ptr = scatter(torch.ones(num_nodes, dtype=torch.long, device=x.device),
                  batch, dim=0, dim_size=num_graphs, reduce='sum').cumsum(0)
    ptr = torch.cat([x.new_zeros(1, dtype=torch.long), ptr])
    
    all_edge_indices = []
    
    for i in range(num_graphs):
        # 1. 提取当前子图的特征和节点全局索引
        start_idx = ptr[i]
        end_idx = ptr[i+1]
        
        # 当前子图的特征 X_i
        x_i = x[start_idx:end_idx]
        n_i = x_i.size(0) # 当前子图的节点数
        
        if n_i == 0:
            continue
            
        # 2. 计算当前子图内的距离矩阵 D_i (n_i x n_i)
        
        if cosine:
            x_norm = x_i / x_i.norm(dim=1, keepdim=True).clamp(min=1e-8)
            S = torch.mm(x_norm, x_norm.t())
            D_i = 1.0 - S
        else:
            x_sq = torch.sum(x_i**2, dim=1, keepdim=True)
            D_i = x_sq + x_sq.t() - 2 * torch.mm(x_i, x_i.t())
            D_i = D_i.clamp(min=0.0)

        # 3. 选取最远邻居 (topk)
        
        k_adjusted = min(k + 1, n_i)
        
        # _, indices: 形状 [n_i, k_adjusted]，包含最远邻居的局部索引
        _, indices = torch.topk(D_i, k=k_adjusted, dim=1, largest=True)
        
        # 4. 构建 edge_index (局部索引)
        source_nodes_local = torch.arange(n_i, device=x.device).repeat_interleave(k_adjusted)
        target_nodes_local = indices.flatten()
        
        edge_index_local = torch.stack([source_nodes_local, target_nodes_local], dim=0)

        # 5. 可选: 移除自环 (局部索引)
        if not loop:
            mask = edge_index_local[0] != edge_index_local[1]
            edge_index_local = edge_index_local[:, mask]
        
        # 6. 转换回全局索引并存储
        # 局部索引 + start_idx = 全局索引
        edge_index_global = edge_index_local + start_idx

        if not direction:
            # 强制双向：拼接翻转后的边
            edge_index_global = torch.cat([edge_index_global, edge_index_global.flip(0)], dim=1)

        all_edge_indices.append(edge_index_global)

    # 7. 合并所有子图的边索引
    if len(all_edge_indices) == 0:
        return x.new_empty((2, 0), dtype=torch.long)
        
    final_edge_index = torch.cat(all_edge_indices, dim=1)
    
    return final_edge_index

def k_random_graph(
    x: Tensor,
    k: int,
    batch: Optional[Tensor] = None,
    loop: bool = False
) -> Tensor:
    """
    构建基于随机选择 K 个邻居的图的边索引 (edge_index)，支持批处理。
    使用向量化采样以提升性能，允许重复邻居但可选是否包含自环。

    参数:
        x (Tensor): 节点特征矩阵，形状为 [num_nodes, num_features]。
        k (int): 每个节点要连接的随机邻居数量。
        batch (Optional[Tensor]): 批次向量，形状为 [num_nodes]。
        loop (bool): 是否允许自环。

    返回:
        Tensor: 边索引，形状为 [2, num_edges]。
    """
    
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    
    num_nodes = x.size(0)
    num_graphs = batch.max().item() + 1

    ptr = scatter(torch.ones(num_nodes, dtype=torch.long, device=x.device),
                  batch, dim=0, dim_size=num_graphs, reduce='sum').cumsum(0)
    ptr = torch.cat([x.new_zeros(1, dtype=torch.long), ptr])

    all_edge_indices = []

    for i in range(num_graphs):
        start_idx = ptr[i]
        end_idx = ptr[i + 1]
        n_i = end_idx - start_idx

        if n_i <= 1:
            continue

        k_sample = k if loop else min(k, n_i - 1)
        if k_sample <= 0:
            continue

        source_nodes_local = torch.arange(n_i, device=x.device)

        # 向量化采样目标节点（允许重复）
        target_nodes_local_matrix = torch.randint(
            low=0, high=n_i, size=(n_i, k_sample), device=x.device
        )

        if not loop:
            # 排除自环：重新采样冲突位置直到无自环
            source_matrix = source_nodes_local.view(-1, 1).expand(-1, k_sample)
            mask = target_nodes_local_matrix == source_matrix
            while mask.any():
                resample = torch.randint(0, n_i, size=mask.shape, device=x.device)
                target_nodes_local_matrix[mask] = resample[mask]
                mask = target_nodes_local_matrix == source_matrix

        target_nodes_local = target_nodes_local_matrix.flatten()
        source_nodes_local_repeat = source_nodes_local.repeat_interleave(k_sample)

        edge_index_local = torch.stack([source_nodes_local_repeat, target_nodes_local], dim=0)
        edge_index_global = edge_index_local + start_idx
        all_edge_indices.append(edge_index_global)

    if len(all_edge_indices) == 0:
        return x.new_empty((2, 0), dtype=torch.long)

    final_edge_index = torch.cat(all_edge_indices, dim=1)
    return final_edge_index