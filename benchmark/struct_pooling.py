import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_scatter import scatter_mean, scatter_max
import math


class StructPool(nn.Module):
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
        super(StructPool, self).__init__()
        
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


class StructPoolingWithGlobal(nn.Module):
    """
    增强版 StructPool，包含全局信息聚合
    """
    
    def __init__(self, in_channels, ratio=0.5, dropout=0.1, negative_slope=0.2):
        super(StructPoolingWithGlobal, self).__init__()
        
        self.in_channels = in_channels
        self.ratio = ratio
        
        # 基础 StructPool
        self.struct_pool = StructPool(
            in_channels, ratio, dropout, negative_slope
        )
        
        # 全局信息聚合
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 获取全局图表示
        global_mean = global_mean_pool(x, batch)
        global_max = global_max_pool(x, batch)
        global_info = torch.cat([global_mean, global_max], dim=1)
        global_info = self.global_proj(global_info)
        
        # 应用 StructPool
        x_pooled, edge_index_pooled, batch_pooled = self.struct_pool(
            x, edge_index, batch
        )
        
        # 将全局信息广播到每个节点
        global_broadcast = global_info[batch_pooled]
        
        # 门控融合
        gate_values = self.gate(torch.cat([x_pooled, global_broadcast], dim=1))
        x_pooled = gate_values * x_pooled + (1 - gate_values) * global_broadcast
        
        return x_pooled, edge_index_pooled, batch_pooled


# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    num_nodes = 10
    in_channels = 16
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6, 9, 8]
    ], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    
    # 测试基础版 StructPool
    struct_pool = StructPool(in_channels=in_channels, ratio=0.6)
    x_pooled, edge_index_pooled, batch_pooled = struct_pool(x, edge_index, batch)
    
    print("基础版 StructPool:")
    print(f"输入节点数: {x.size(0)}")
    print(f"池化后节点数: {x_pooled.size(0)}")
    print(f"输入边数: {edge_index.size(1)}")
    print(f"池化后边数: {edge_index_pooled.size(1)}")
    print(f"输入批次: {batch.unique_consecutive()}")
    print(f"池化后批次: {batch_pooled.unique_consecutive()}")
    
    # 测试增强版 StructPool
    struct_pool_global = StructPoolingWithGlobal(in_channels=in_channels, ratio=0.6)
    x_pooled_global, edge_index_pooled_global, batch_pooled_global = struct_pool_global(
        x, edge_index, batch
    )
    
    print("\n增强版 StructPool:")
    print(f"池化后节点数: {x_pooled_global.size(0)}")
    print(f"池化后边数: {edge_index_pooled_global.size(1)}")