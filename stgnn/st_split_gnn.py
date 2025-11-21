import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from numpy import ndarray
from torch_geometric.nn import GCNConv, GINConv, GraphNorm, global_mean_pool
from torch.nn import MultiheadAttention

def kruskal_mst_track_usage(edge_index: Tensor, edge_weight: ndarray, num_nodes: int):
    # Step 1: 转为 numpy 并排序
    edge_index = edge_index.cpu().numpy()
    original_idx = np.arange(len(edge_weight))
    sorted_idx = np.argsort(edge_weight)
    edge_index = edge_index[:, sorted_idx]
    edge_weight_sorted = edge_weight[sorted_idx]
    original_idx_sorted = original_idx[sorted_idx]

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

    # Step 3: 构建 MST 并记录使用的边
    mst_edges = []
    used_edge_indices = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        if union(u, v):
            mst_edges.append([u, v])
            used_edge_indices.append(i)
            if len(mst_edges) == num_nodes - 1:
                break

    # Step 4: 更新被使用边的权重
    edge_weight_sorted[used_edge_indices] += 1

    # Step 5: 还原为原始顺序
    edge_weight_updated = np.empty_like(edge_weight_sorted)
    edge_weight_updated[original_idx_sorted] = edge_weight_sorted

    # Step 6: 返回结果
    mst_edge_index = torch.tensor(mst_edges, dtype=torch.long).t().contiguous()
    return mst_edge_index, edge_weight_updated

def split_graph(x: Tensor, edge_index: Tensor, num_splits: int):
    edge_weight = np.zeros(edge_index.shape[1], dtype=np.int32)
    edge_index_splited = []

    for i in range(num_splits):
        mst_edge_index, edge_weight = kruskal_mst_track_usage(edge_index, edge_weight, x.size(0))

        # 补全为双向边
        reversed_edge_index = mst_edge_index[[1, 0], :]  # 交换行得到反向边
        bidir_edge_index = torch.cat([mst_edge_index, reversed_edge_index], dim=1)

        edge_index_splited.append(bidir_edge_index)

    return edge_index_splited

class GNNs(nn.Module):
    def __init__(self, channels: int = 128, num_layers: int = 3, backbone: str = 'GCN', use_graph_norm: bool = True, dropout = 0.5):
        super(GNNs, self).__init__()
        self.channel = max(channels, 1)
        self.num_layers = num_layers
        self.backbone = str.upper(backbone)
        self.convs = self._build_conv()
        self.use_graph_norm = use_graph_norm
        self.dropout = dropout
        if use_graph_norm:
            self.graph_norm = self._build_graph_norm()

    def _build_conv(self):
        convs = nn.ModuleList()
        if self.backbone == 'GCN':
            for _ in range(self.num_layers):
                convs.append(GCNConv(self.channel, self.channel))
        return convs
    
    def _build_graph_norm(self):
        norms = nn.ModuleList()
        for _ in range(self.num_layers):
            norms.append(GraphNorm(self.channel))
        return norms

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        x_history = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if self.use_graph_norm:
                x = self.graph_norm[i](x, batch)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_history.append(x)
        return x, x_history

class SpanTreeSplitGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 3, dropout: float = 0.5, num_splits: int = 4):
        super(SpanTreeSplitGNN, self).__init__()
        self.num_splits = num_splits
        self.in_channels = max(1, in_channels)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.main_gnn = GNNs(channels=self.hidden_channels, num_layers=self.num_layers, dropout=self.dropout)
        self.local_struct_gnns = self._build_gnns(1)
        self.embedding = self._build_embedding()
        self.dense_merge = nn.Linear(self.hidden_channels * (self.num_splits + 1), self.hidden_channels)
        # self.alpha = nn.Parameter(torch.ones(self.hidden_channels))
        # self.attention = MultiheadAttention(embed_dim=self.hidden_channels, num_heads=1, dropout=dropout)

    def _build_gnns(self, num: int):
        gnns = nn.ModuleList()
        for _ in range(num):
            gnns.append(GNNs(channels=self.hidden_channels, num_layers=self.num_layers, dropout=self.dropout))
        return gnns
    
    def _build_embedding(self):
        return nn.Linear(in_features=self.in_channels, out_features=self.hidden_channels)
    
    # def cross_attention(self, main_x: Tensor, local_struct_x: Tensor, batch: Tensor):
    #     outputs = []
    #     for i in batch.unique():
    #         mask = batch == i
    #         span_mask = batch.repeat_interleave(self.num_splits) == i
    #         main_x_i = main_x[mask].unsqueeze(1)  # (N_i, 1, C)
    #         local_x_i = local_struct_x[span_mask].unsqueeze(1)  # (M_i, 1, C)

    #         attn_output, _ = self.attention(main_x_i, local_x_i, local_x_i)
    #         outputs.append(attn_output.squeeze(1))  # (N_i, C)
    #     return torch.cat(outputs, dim=0)

    # def cross_attention(self, main_x: Tensor, local_struct_x: Tensor, batch: Tensor):
    #     from torch_geometric.utils import to_dense_batch
    #     # 构造 span_batch：每个节点复制 num_splits 次
    #     span_batch = batch.repeat_interleave(self.num_splits)

    #     # dense 化：每个图变成一个 [max_nodes, C] 的矩阵
    #     q, q_mask = to_dense_batch(main_x, batch)                  # [B, L_q, C]
    #     kv, kv_mask = to_dense_batch(local_struct_x, span_batch)   # [B, L_kv, C]

    #     # 转为 (L, B, C) 以适配 MultiheadAttention（默认 batch_first=False）
    #     q = q.transpose(0, 1)   # [L_q, B, C]
    #     k = kv.transpose(0, 1)  # [L_kv, B, C]
    #     v = kv.transpose(0, 1)  # [L_kv, B, C]

    #     # 构造 mask：True 表示要 mask 掉
    #     key_padding_mask = ~kv_mask  # [B, L_kv]

    #     # 一次性计算所有 attention
    #     attn_output, _ = self.attention(q, k, v, key_padding_mask=key_padding_mask)  # [L_q, B, C]

    #     # 转回 [B, L_q, C]，再去掉 padding
    #     attn_output = attn_output.transpose(0, 1)  # [B, L_q, C]
    #     out = attn_output[q_mask]                 # [N, C]

    #     return out

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        ori_edge_index = edge_index
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)  # 生成一个随机排列
        edge_index = edge_index[:, perm]  # 在边维度上打乱

        ori_x = x
        x = self.embedding(x)
        
        _, main_x_history = self.main_gnn(x, ori_edge_index, batch)
        
        with torch.no_grad():
            edge_index_splited = split_graph(x, edge_index, self.num_splits)
            edge_index_splited.insert(0, edge_index)  # 添加原始图作为第一个子图
            edge_index_splited = list(map(lambda e: e.to(x.device), edge_index_splited))
    
        xs = []
        for i in range(self.num_splits):
            edge_index_i = edge_index_splited[i]
            _, x_history = self.local_struct_gnns[0](x, edge_index_i, batch)
            xs.append(x_history)

        x_layers = []
        g_layers = []
        for l in range(self.num_layers):
            x_ls = [xs[s][l] for s in range(self.num_splits)]
            # x_l = torch.sum(torch.stack(x_ls), dim=0)
            x_ls.append(main_x_history[l])

            x_l = torch.cat(x_ls, dim=1)
            x_l = self.dense_merge(x_l)
            
            # x_l = torch.cat(x_ls, dim=0)
            # x_l = self.cross_attention(main_x_history[l], x_l, batch) + main_x_history[l]

            x_layers.append(x_l)
            g_layers.append(global_mean_pool(x_l, batch))

        # x_all = torch.mean(torch.stack(x_layers), dim=0)
        graph_all = torch.mean(torch.stack(g_layers), dim=0)

        return graph_all, 0