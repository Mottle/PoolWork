import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph, softmax
from torch_scatter import scatter_mean

class EdgeNodeAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, max_clusters):
        super().__init__()
        self.max_clusters = max_clusters
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_proj = nn.Linear(in_channels, hidden_channels)
        self.att_proj = nn.Linear(hidden_channels, max_clusters)

    def forward(self, x, edge_index, edge_attr=None):
        num_edges = edge_index.size(1)
        x_proj = self.node_proj(x)

        if edge_attr is None:
            edge_attr = torch.zeros(num_edges, x.size(1), device=x.device)
        edge_proj = self.edge_proj(edge_attr)

        row, col = edge_index
        x_i = x_proj[col]
        x_j = x_proj[row]
        h = x_i + x_j + edge_proj

        att_score = self.att_proj(h)
        att_score = softmax(att_score, col)

        return att_score  # [E, K]

class ENAHPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, max_clusters):
        super().__init__()
        self.max_clusters = max_clusters
        self.att_layer = EdgeNodeAttention(in_channels, hidden_channels, max_clusters)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        device = x.device
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        if edge_attr is None:
            edge_feat_dim = self.att_layer.edge_proj.in_features
            edge_attr = torch.zeros(edge_index.size(1), edge_feat_dim, device=device)

        outputs = []
        for b in batch.unique():
            mask = batch == b
            node_idx = mask.nonzero(as_tuple=False).view(-1)
            num_nodes = x.size(0)

            edge_index_b, edge_attr_b = subgraph(
                node_idx,
                edge_index,
                edge_attr,
                relabel_nodes=True,
                num_nodes=num_nodes
            )

            x_b = x[mask]
            num_nodes_b = x_b.size(0)
            num_clusters_b = min(self.max_clusters, num_nodes_b)

            if edge_index_b.size(1) == 0:
                # 构造空邻接矩阵
                adj_sparse = torch.zeros((num_nodes_b, num_nodes_b), device=device)
            else:
                # 构造稀疏邻接矩阵
                adj_sparse = torch.zeros((num_nodes_b, num_nodes_b), device=device)
                adj_sparse[edge_index_b[0], edge_index_b[1]] = 1.0

            att_scores_full = self.att_layer(x_b, edge_index_b, edge_attr_b)
            att_scores = att_scores_full[:, :num_clusters_b]

            assign_score = scatter_mean(att_scores, edge_index_b[1], dim=0, dim_size=num_nodes_b)
            S = F.softmax(assign_score, dim=-1)  # [n, K]

            x_pooled = torch.matmul(S.T, x_b)  # [K, D]
            adj_pooled = torch.matmul(S.T, torch.matmul(adj_sparse, S))  # [K, K]

            edge_index_pooled, edge_weight_pooled = adj_pooled.nonzero(as_tuple=True), adj_pooled[adj_pooled != 0]
            edge_index_pooled = torch.stack(edge_index_pooled, dim=0)

            batch_pooled = torch.full((num_clusters_b,), b, device=device)
            outputs.append((x_pooled, edge_index_pooled, edge_weight_pooled, batch_pooled))

        x_all, ei_all, ew_all, batch_all = zip(*outputs)
        x_out = torch.cat(x_all, dim=0)
        edge_index_out = torch.cat(ei_all, dim=1)
        edge_weight_out = torch.cat(ew_all, dim=0)
        batch_out = torch.cat(batch_all, dim=0)

        return x_out, edge_index_out, edge_weight_out, batch_out
