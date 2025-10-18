import context
import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import coalesce
from mambo.linize import convert_to_line_graph_with_batch_vectorized_clean

class MamboPoolingWithEdgeGraphScore(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, gnn_layers=2, dropout=0.0):
        super().__init__()
        # 边得分计算GNN
        self.edge_scoring_gnn = EdgeScoringGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=gnn_layers,
            dropout=dropout
        )
        
        # 新节点特征生成网络（可选改进）
        self.node_feature_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels)
        )
        
        self.in_channels = in_channels
        self.logging_enable = False
    
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            batch: 批处理索引 [num_nodes]
            
        Returns:
            x_pool: 池化后节点特征
            edge_index_pool: 池化后边索引
            batch_pool: 池化后批处理索引
            edge_scores: 边得分
            perm: 节点重排索引
        """
        from time import perf_counter

        start_time = perf_counter()
        # 1. 转换为边图
        # 1. 转换为边图
        line_data = convert_to_line_graph_with_batch_vectorized_clean(
            x, edge_index, batch, logging_enable=self.logging_enable
        )
        end_time = perf_counter()
        convert_time = end_time - start_time
        
        start_time = perf_counter()
        # 2. 使用GNN计算边得分
        edge_scores = self.edge_scoring_gnn(line_data.x, line_data.edge_index, line_data.batch)
        end_time = perf_counter()
        scoring_time = end_time - start_time
        
        start_time = perf_counter()
        # 3. 选择top-k边进行池化
        k = min(edge_scores.size(0) // 2, x.size(0) // 2)  # 至少保留一半节点
        if k == 0:
            k = 1
        
        topk_scores, topk_indices = torch.topk(edge_scores, k)
        end_time = perf_counter()
        topk_time = end_time - start_time
        
        if self.logging_enable:
            print(f"选择top-{k}边进行池化, 得分范围: {topk_scores.min().item():.4f} ~ {topk_scores.max().item():.4f}")
        
        start_time = perf_counter()
        # 4. 执行边收缩
        x_pool, edge_index_pool, batch_pool, perm = self.contract_edges(
            x, edge_index, batch, line_data.original_edge_index, topk_indices, topk_scores
        )
        end_time = perf_counter()
        contract_time = end_time - start_time

        all_time = convert_time + scoring_time + topk_time + contract_time

        if self.logging_enable:
            print(f"转换时间: {convert_time:.4f}s, 计算边得分时间: {scoring_time:.4f}s, 选择top-k边时间: {topk_time:.4f}s, 边收缩时间: {contract_time:.4f}s, 总时间: {all_time:.4f}s")
            #占比
            print(f"转换时间占比: {convert_time / all_time:.4f}, 计算边得分时间占比: {scoring_time / all_time:.4f}, 选择top-k边时间占比: {topk_time / all_time:.4f}, 边收缩时间占比: {contract_time / all_time:.4f}")
        
        return x_pool, edge_index_pool, batch_pool, edge_scores, perm
    
    def contract_edges(self, x, edge_index, batch, clean_edge_index, topk_indices, topk_scores):
        """
        边收缩操作
        """
        num_nodes = x.size(0)
        device = x.device
        
        # 获取被选中的边
        selected_edges = clean_edge_index[:, topk_indices]  # [2, k]
        
        # 创建节点映射：每个节点映射到新的超级节点索引
        node_to_cluster = torch.arange(num_nodes, device=device)
        
        # 对于每条被选中的边，将两个节点映射到同一个簇
        # 我们选择源节点作为簇的代表
        src_nodes = selected_edges[0]
        dst_nodes = selected_edges[1]
        
        # 更新映射：目标节点映射到源节点
        for i in range(src_nodes.size(0)):
            node_to_cluster[dst_nodes[i]] = src_nodes[i]
        
        # 找到所有唯一的簇代表
        unique_clusters = torch.unique(node_to_cluster)
        num_new_nodes = unique_clusters.size(0)
        
        # 创建从旧节点到新节点的映射
        old_to_new = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        old_to_new[unique_clusters] = torch.arange(num_new_nodes, device=device)
        
        # 应用映射
        new_indices = old_to_new[node_to_cluster]
        
        # 生成新节点特征（使用得分加权的特征聚合）
        x_pool = self.aggregate_node_features(x, selected_edges, topk_scores, new_indices, num_new_nodes)
        
        # 生成新的批处理索引
        batch_pool = batch[unique_clusters]
        
        # 生成新的边索引（避免重复边和自环）
        edge_index_pool = self.build_new_edges(edge_index, new_indices, batch)
        
        return x_pool, edge_index_pool, batch_pool, new_indices
    
    def aggregate_node_features(self, x, selected_edges, edge_scores, new_indices, num_new_nodes):
        """
        聚合节点特征生成新节点特征
        """
        device = x.device
        x_pool = torch.zeros(num_new_nodes, self.in_channels, device=device)
        
        # 对于每个新节点（簇）
        for new_idx in range(num_new_nodes):
            # 找到属于这个簇的所有旧节点
            old_indices = torch.where(new_indices == new_idx)[0]
            
            if len(old_indices) == 1:
                # 单节点簇，直接复制特征
                x_pool[new_idx] = x[old_indices[0]]
            else:
                # 多节点簇，需要聚合
                # 找到连接这些节点的边
                cluster_edges_mask = torch.isin(selected_edges[0], old_indices) & torch.isin(selected_edges[1], old_indices)
                cluster_edge_scores = edge_scores[torch.where(cluster_edges_mask)[0]]
                
                if len(cluster_edge_scores) > 0:
                    # 使用边得分加权的特征聚合
                    weights = cluster_edge_scores.unsqueeze(1)
                    cluster_x = x[old_indices]
                    # 简单平均（可以改进为更复杂的加权方式）
                    x_pool[new_idx] = cluster_x.mean(dim=0)
                else:
                    # 没有连接边，使用平均
                    x_pool[new_idx] = x[old_indices].mean(dim=0)
        
        return x_pool
    
    def build_new_edges(self, old_edge_index, new_indices, batch):
        """
        构建新的边索引
        """
        if old_edge_index.size(1) == 0:
            return old_edge_index
        
        # 映射旧边到新节点
        new_src = new_indices[old_edge_index[0]]
        new_dst = new_indices[old_edge_index[1]]
        
        # 移除自环
        non_self_loop_mask = (new_src != new_dst)
        new_src = new_src[non_self_loop_mask]
        new_dst = new_dst[non_self_loop_mask]
        
        if new_src.size(0) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=old_edge_index.device)
        
        # 合并重复边
        new_edge_index = torch.stack([new_src, new_dst], dim=0)
        new_edge_index, _ = coalesce(new_edge_index, None, new_indices.max().item() + 1, new_indices.max().item() + 1)
        
        return new_edge_index
    
    def enable_logging(self):
        """启用日志输出"""
        self.logging_enable = True
        self.edge_scoring_gnn.logging_enable = True
    
    def disable_logging(self):
        """禁用日志输出"""
        self.logging_enable = False
        self.edge_scoring_gnn.logging_enable = False

class EdgeScoringGNN(nn.Module):
    """
    使用GNN在边图上计算边得分的模块
    """
    def __init__(self, in_channels, hidden_channels = 64, out_channels=1, num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        # 输入层
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # 输出层
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, out_channels))
        
        # 输出激活函数
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = line_data.x, line_data.edge_index, line_data.batch
        
        # 多层GNN
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        x = self.convs[-1](x, edge_index)
        
        # 应用sigmoid得到边得分
        edge_scores = self.sigmoid(x).squeeze(-1)
        
        return edge_scores