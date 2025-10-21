import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GCNConv
from torch_geometric.utils import to_dense_adj, degree, dense_to_sparse
import math


class BadStructPool(nn.Module):
    """
    StructPool: 结构化图池化层
    基于节点特征和图结构信息进行池化，保留图的重要结构特性
    """
    
    def __init__(self, in_channels, ratio=0.5, dropout=0.1, negative_slope=0.2):
        """
        初始化 StructPool
        
        Args:
            in_channels (int): 输入特征维度
            ratio (float): 池化比率，保留节点的比例 (0-1)
            dropout (float): Dropout比率
            negative_slope (float): LeakyReLU负斜率
        """
        super(BadStructPool, self).__init__()
        
        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        
        # 结构感知的投影层
        self.struct_proj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 1)
        )
        
        # 特征重要性投影层
        self.feat_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1)
        )
        
        # 自适应权重参数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        # 池化后特征变换
        self.post_pool = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.struct_proj[0].weight, gain=1.414)
        nn.init.xavier_uniform_(self.struct_proj[3].weight, gain=1.414)
        nn.init.xavier_uniform_(self.feat_proj[0].weight, gain=1.414)
        nn.init.xavier_uniform_(self.feat_proj[3].weight, gain=1.414)
        nn.init.xavier_uniform_(self.post_pool[0].weight, gain=1.414)
        nn.init.constant_(self.struct_proj[0].bias, 0)
        nn.init.constant_(self.struct_proj[3].bias, 0)
        nn.init.constant_(self.feat_proj[0].bias, 0)
        nn.init.constant_(self.feat_proj[3].bias, 0)
        nn.init.constant_(self.post_pool[0].bias, 0)
    
    def forward(self, x, edge_index, batch=None, return_attention_weights=False):
        """
        前向传播
        
        Args:
            x (Tensor): 节点特征 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            batch (Tensor, optional): 批次向量 [N]
            return_attention_weights (bool): 是否返回注意力权重
            
        Returns:
            x_pooled (Tensor): 池化后节点特征
            edge_index_pooled (Tensor): 池化后边索引
            batch_pooled (Tensor): 池化后批次向量
            attention_weights (Tensor, optional): 注意力权重
        """
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        num_nodes = x.size(0)
        k = max(1, int(self.ratio * num_nodes))
        
        # 1. 计算节点重要性分数
        importance_scores = self._compute_importance_scores(x, edge_index, batch)
        
        # 2. 选择top-k重要节点
        topk_scores, topk_indices = self._select_topk_nodes(importance_scores, k, batch)
        
        # 3. 池化节点特征
        x_pooled = x[topk_indices]
        
        # 4. 构建池化后的边索引
        edge_index_pooled = self._construct_pooled_edges(
            edge_index, topk_indices, num_nodes
        )
        
        # 5. 更新批次向量
        batch_pooled = batch[topk_indices]
        
        # 6. 应用特征变换
        x_pooled = self.post_pool(x_pooled)
        
        if return_attention_weights:
            # 创建注意力权重矩阵
            attention_weights = torch.zeros(num_nodes, device=x.device)
            attention_weights[topk_indices] = topk_scores
            return x_pooled, edge_index_pooled, batch_pooled, attention_weights
        
        return x_pooled, edge_index_pooled, batch_pooled
    
    def _compute_importance_scores(self, x, edge_index, batch):
        """
        计算节点重要性分数
        结合结构信息和特征信息
        """
        # 计算结构信息（节点度）
        row, col = edge_index
        node_degree = degree(row, x.size(0), dtype=x.dtype)
        node_degree = node_degree.unsqueeze(1)
        
        # 归一化度特征
        degree_norm = F.normalize(node_degree, p=2, dim=0)
        
        # 计算结构感知分数
        struct_input = torch.cat([x, degree_norm.expand_as(x)], dim=1)
        struct_scores = self.struct_proj(struct_input).squeeze()
        
        # 计算特征重要性分数
        feat_scores = self.feat_proj(x).squeeze()
        
        # 自适应权重组合
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        weights_sum = alpha + beta
        alpha = alpha / weights_sum
        beta = beta / weights_sum
        
        importance_scores = alpha * struct_scores + beta * feat_scores
        
        return importance_scores
    
    def _select_topk_nodes(self, scores, k, batch):
        """选择每个图中top-k重要节点"""
        batch_size = batch.max().item() + 1
        selected_indices = []
        selected_scores = []
        
        for i in range(batch_size):
            mask = (batch == i)
            graph_scores = scores[mask]
            graph_indices = torch.where(mask)[0]
            
            # 选择当前图中的top-k节点
            if len(graph_scores) <= k:
                # 如果图中节点数小于k，选择所有节点
                topk_indices = graph_indices
                topk_scores = graph_scores
            else:
                # 选择top-k节点
                topk_values, topk_local_indices = torch.topk(
                    graph_scores, k, sorted=False
                )
                topk_indices = graph_indices[topk_local_indices]
                topk_scores = topk_values
            
            selected_indices.append(topk_indices)
            selected_scores.append(topk_scores)
        
        # 合并所有批次的结果
        selected_indices = torch.cat(selected_indices)
        selected_scores = torch.cat(selected_scores)
        
        return selected_scores, selected_indices
    
    def _construct_pooled_edges(self, edge_index, selected_indices, num_nodes):
        """
        构建池化后的边索引
        只保留连接两个被选择节点的边
        """
        # 创建节点映射表
        node_mapping = -torch.ones(num_nodes, dtype=torch.long, device=edge_index.device)
        node_mapping[selected_indices] = torch.arange(
            len(selected_indices), device=edge_index.device
        )
        
        # 过滤边，只保留两个端点都被选择的边
        row, col = edge_index
        mask = (node_mapping[row] >= 0) & (node_mapping[col] >= 0)
        edge_index_pooled = torch.stack([
            node_mapping[row[mask]], node_mapping[col[mask]]
        ], dim=0)
        
        return edge_index_pooled
    
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'ratio={self.ratio})')

# ℓ-hop 邻接矩阵构建
def compute_l_hop_adj(adj, l=2):
    adj_l = adj.clone()
    for _ in range(l - 1):
        adj_l = torch.matmul(adj_l, adj)
    adj_l = (adj_l > 0).float()
    return adj_l

# 高斯核模块
class GaussianKernel(nn.Module):
    def __init__(self, in_channels, num_kernels=2):
        super().__init__()
        self.means = nn.Parameter(torch.randn(num_kernels, in_channels))
        self.scales = nn.Parameter(torch.ones(num_kernels, in_channels))
        self.weights = nn.Parameter(torch.ones(num_kernels))

    def forward(self, f_i, f_j):
        # f_i, f_j: [N, D]
        N = f_i.size(0)
        K = self.means.size(0)

        f_i_exp = f_i.unsqueeze(1).expand(N, K, -1)
        f_j_exp = f_j.unsqueeze(1).expand(N, K, -1)

        diff_i = (f_i_exp - self.means) / self.scales
        diff_j = (f_j_exp - self.means) / self.scales

        dist_i = -torch.sum(diff_i ** 2, dim=-1)  # [N, K]
        dist_j = -torch.sum(diff_j ** 2, dim=-1)  # [N, K]

        kernel = dist_i.unsqueeze(1) + dist_j.unsqueeze(0)  # [N, N, K]
        kernel = torch.exp(kernel)  # 高斯核值
        weighted = torch.matmul(kernel, self.weights)  # [N, N]
        return weighted

# 标签兼容性矩阵
class CompatibilityMatrix(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(num_clusters, num_clusters))

    def forward(self, Q):
        return torch.matmul(Q, self.mu)  # [N, C]

# Mean-field 推理
class MeanFieldCRF(nn.Module):
    def __init__(self, num_clusters, num_iter=5, sharpen_temp=0.5):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_iter = num_iter
        self.sharpen_temp = sharpen_temp
        self.compatibility = CompatibilityMatrix(num_clusters)

    def forward(self, unary, pairwise_weight):
        Q = F.softmax(unary, dim=-1)  # [N, C]
        for _ in range(self.num_iter):
            compat = self.compatibility(Q)  # [N, C]
            message = torch.matmul(pairwise_weight, compat)  # [N, C]
            Q = unary + message
            Q = F.softmax(Q / self.sharpen_temp, dim=-1)
        return Q

# StructPool 主模块
class StructPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters, num_iter=5, l_hop=2, num_kernels=2):
        super().__init__()
        self.encoder = GCNConv(in_channels, hidden_channels)
        self.unary = nn.Linear(hidden_channels, num_clusters)
        self.kernel = GaussianKernel(hidden_channels, num_kernels)
        self.crf = MeanFieldCRF(num_clusters, num_iter)
        self.l_hop = l_hop
        self.num_clusters = num_clusters

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.encoder(x, edge_index))  # [N, D]

        outputs = []
        for b in batch.unique():
            mask = batch == b
            if mask.sum() == 0:
                continue  # 跳过空图

            x_b = x[mask]
            if x_b.size(0) == 0:
                continue  # 跳过无节点图

            edge_index_b = edge_index[:, (mask[edge_index[0]] & mask[edge_index[1]])]
            if edge_index_b.size(1) == 0:
                # 构造一个空邻接矩阵
                adj_b = torch.zeros((x_b.size(0), x_b.size(0)), device=x.device)
            else:
                node_offset = mask.nonzero()[0].item()
                edge_index_b = edge_index_b - node_offset
                adj_b = to_dense_adj(edge_index_b)[0]
            adj_l = compute_l_hop_adj(adj_b, self.l_hop)  # ℓ-hop 邻接

            unary_b = self.unary(x_b)  # [n, C]
            pairwise_b = self.kernel(x_b, x_b) * adj_l  # [n, n]

            assign_b = self.crf(unary_b, pairwise_b)  # [n, C]
            x_pooled = torch.matmul(assign_b.T, x_b)  # [C, D]
            adj_pooled = torch.matmul(assign_b.T, torch.matmul(adj_b, assign_b))  # [C, C]
            edge_index_pooled, edge_weight_pooled = dense_to_sparse(adj_pooled)

            batch_pooled = torch.full((self.num_clusters,), b, device=x.device)
            outputs.append((x_pooled, edge_index_pooled, edge_weight_pooled, batch_pooled))

        x_all, ei_all, ew_all, batch_all = zip(*outputs)
        x_out = torch.cat(x_all, dim=0)
        edge_index_out = torch.cat(ei_all, dim=1)
        edge_weight_out = torch.cat(ew_all, dim=0)
        batch_out = torch.cat(batch_all, dim=0)

        return x_out, edge_index_out, edge_weight_out, batch_out