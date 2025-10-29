import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_max_pool, global_add_pool, GraphNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_geometric.data import Data
import networkx as nx
from perf_counter import get_time_sync

# def kruskal_mst(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
#     # Step 1: 按边权从大到小排序
#     sorted_idx = torch.argsort(edge_weight, descending=True)
#     edge_index = edge_index[:, sorted_idx]

#     # Step 2: 初始化并查集
#     parent = torch.arange(num_nodes, device=edge_index.device)

#     # Step 3: 初始化输出掩码
#     mst_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
#     edge_count = 0

#     # Step 4: 张量化 find + union
#     def find(u):
#         while True:
#             parent_u = parent[u]
#             grandparent_u = parent[parent_u]
#             if torch.equal(parent_u, grandparent_u):
#                 break
#             parent[u] = grandparent_u
#             u = grandparent_u
#         return parent[u]

#     for i in range(edge_index.size(1)):
#         u = edge_index[0, i]
#         v = edge_index[1, i]
#         pu = find(u)
#         pv = find(v)
#         if pu != pv:
#             parent[pu] = pv
#             mst_mask[i] = True
#             edge_count += 1
#             if edge_count == num_nodes - 1:
#                 break

#     return edge_index[:, mst_mask]

def kruskal_mst(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
    # 转为 numpy 加速处理
    edge_index = edge_index.cpu().numpy()
    edge_weight = edge_weight.cpu().numpy()

    # Step 1: 按边权从大到小排序
    sorted_idx = np.argsort(-edge_weight)
    edge_index = edge_index[:, sorted_idx]

    # Step 2: 初始化并查集
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=int)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        return True

    # Step 3: 构建 MST
    mst_edges = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        if union(u, v):
            mst_edges.append([u, v])
            if len(mst_edges) == num_nodes - 1:
                break

    # Step 4: 转回 PyTorch 张量
    mst_edge_index = torch.tensor(mst_edges, dtype=torch.long).t().contiguous()
    return mst_edge_index

class SpanTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0.5):
        super(SpanTreeConv, self).__init__()
        self.edge_score_conv = nn.Linear(in_channels, 1)
        self.mst_conv = GCNConv(in_channels, out_channels)
        # self.edge_attr_conv = nn.Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        edge_score = self.edge_score_conv(edge_attr).squeeze(-1)
        edge_score = F.dropout(edge_score, self.dropout, training=self.training)
        edge_weight = F.softmax(edge_score, dim=0)  # 所有边归一化

        edge_attr = edge_attr * edge_weight.view(-1, 1)

        # t0 = get_time_sync()
        with torch.no_grad():
            span_tree_edge_index = kruskal_mst(edge_index.to('cpu'), edge_weight.to('cpu'), num_nodes=x.size(0)).to(x.device)
        # t1 = get_time_sync()
        # print(f"mst time: {t1 - t0}")

        # 初始化一个与 x 同形状的张量，用于累加边特征
        edge_agg = torch.zeros_like(x)
        # 将边特征加到源节点
        edge_agg.index_add_(0, edge_index[0], edge_attr)
        # 将边特征加到目标节点
        edge_agg.index_add_(0, edge_index[1], edge_attr)
        x = x + edge_agg

        # span_tree_edge_index = to_undirected(span_tree_edge_index)
        mst_x = self.mst_conv(x, span_tree_edge_index)
        mst_x = F.leaky_relu(mst_x)

        # 将边特征加到节点特征上
        # edge_agg = torch.zeros_like(x).to(x.device)
        # edge_agg.index_add_(0, edge_index[0], edge_attr)
        # edge_agg.index_add_(0, edge_index[1], edge_attr)

        # return mst_x + edge_agg
        return mst_x
    
class SpanTreeGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3, dropout = 0.5):
        super(SpanTreeGNN, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        self.build_convs()
        self.build_mst_convs()
        self.build_edge_attr_convs()
        self.build_graph_norms()
        self.merge_linear = nn.Linear(in_features=2 * hidden_channels, out_features=hidden_channels)

    def build_convs(self):
        if self.num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(GCNConv(self.in_channels, self.hidden_channels))
            else:
                self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))

    def build_graph_norms(self):
        self.norms = nn.ModuleList()
        self.mst_norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.norms.append(GraphNorm(self.hidden_channels))
            self.mst_norms.append(GraphNorm(self.hidden_channels))

    def build_mst_convs(self):
        self.mst_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.mst_convs.append(SpanTreeConv(self.hidden_channels, self.hidden_channels, dropout=self.dropout))

    def build_edge_attr_convs(self):
        # self.edge_attr_convs = nn.ModuleList()
        # for i in range(self.num_layers):
        #     self.edge_attr_convs.append(nn.Linear(self.hidden_channels * 2, self.hidden_channels))
        self.edge_attr_convs = nn.Linear(self.hidden_channels * 2, self.hidden_channels)

    def compute_edge_attr(self, x, edge_index, i):
        # 获取源节点和目标节点的特征
        src_x = x[edge_index[0]]  # [E, F]
        dst_x = x[edge_index[1]]  # [E, F]

        # 拼接为边特征
        edge_attr = torch.cat([src_x, dst_x], dim=1)  # [E, 2F]
        # edge_attr = self.edge_attr_convs[i](edge_attr)  # [E, F]
        edge_attr = self.edge_attr_convs(edge_attr)  # [E, F]
        edge_attr = F.dropout(edge_attr, p = self.dropout, training=self.training)
        edge_attr = F.leaky_relu(edge_attr)
        return edge_attr
    
    def forward(self, x, edge_index, batch):
        ori_x = x
        merged_all = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            edge_attr = self.compute_edge_attr(x, edge_index, i)

            mst_x = self.mst_convs[i](x, edge_index, edge_attr)
            mst_x = self.mst_norms[i](x, batch)
            mst_x = F.leaky_relu(mst_x)
            
            graph_feature = global_mean_pool(x, batch)
            mst_feature = global_mean_pool(mst_x, batch)
            merged = torch.cat([graph_feature, mst_feature], dim=1)
            merged = self.merge_linear(merged)
            merged = F.dropout(merged, p=self.dropout, training=self.training)
            merged = F.leaky_relu(merged)
            merged_all.append(merged)
        merge_feature = torch.mean(torch.stack(merged_all, dim=0), dim=0)
        # merge_feature = torch.sum(torch.stack(merged_all, dim=0), dim=0)

        return merge_feature, 0
