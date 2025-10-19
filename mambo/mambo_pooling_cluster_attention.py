import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import remove_self_loops, softmax, coalesce, scatter
from typing import Callable, List, NamedTuple, Optional, Tuple
from torch import Tensor

#先使用GNN学习分配矩阵 S: V -> V', 分配出V'个聚类中心，
#再使用GAT计算出边权重得分，随后应用edge pooling方法完成聚类
#在聚类特征计算部分，使用注意力学习每个节点对聚类中心节点的贡献度
class MamboPoolingWithClusterAttention(nn.Module):
    def __init__(self, input_dim: int, ratio: float = 0.5, cluster_center_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.ratio = ratio
        self.cluster_center_heads = cluster_center_heads

        if ratio > 1 or ratio <= 0:
            raise ValueError("Ratio must be between 0 and 1")
        
        self._build_cluster_center_attention()

    #节点对全局特征的注意力，用于后续TopK选择分配中心
    def _build_cluster_center_attention(self):
        self.cluster_center_attention = MultiheadAttention(embed_dim=self.input_dim, num_heads=self.cluster_center_heads, dropout=0.2)

    def forward(self, x, edge_index, batch):
        cluster_center_attn = self._compute_cluster_center_attention(x, batch)


    def _compute_cluster_center_attention(self, x, batch):
        graph_feature = global_mean_pool(x=x, batch=batch)
        batch_siez = torch.unique(batch).size(0)
        att = []
        for batch_index in range(batch_siez):
            mask = batch == batch_index
            batch_x = x[mask]
            batch_graph_feature = graph_feature[batch_index]
            cluster_center_attention, _ = self.cluster_center_attention(batch_x, batch_graph_feature, batch_graph_feature)
            att.append(cluster_center_attention)
        return torch.cat(att, dim=0)
            


    def _compute_cluster_features(self, x, edge_index):
        # 使用GNN学习分配矩阵 S: V -> V'
        x = self.cluster_gnn(x, edge_index)
        x = F.leaky_relu(x)
        #逐向量切分
        S = x.view(-1, self.k)

        return S
    
    def _compute_edge_scores(self, x, edge_index):
        # 使用GAT计算边权重得分
        x_attn, (edge_index_attn, attention_weights) = self.gat_conv(x, edge_index, return_attention_weights=True)
        if self.gat_heads > 1:
            if self.gat_concat:
                e = self.attention_reduce(attention_weights).squeeze(-1)
            else:
                e = attention_weights.mean(dim=1)
        else:
            e = attention_weights.squeeze(-1)
        # 移除自环
        modified_edge_index, e = remove_self_loops(edge_index_attn, e)
        e = F.dropout(e, p=self.dropout, training=self.training)
        edge_score = self.compute_edge_score_softmax(e, edge_index, x.size(0))

        return x_attn, modified_edge_index, edge_score