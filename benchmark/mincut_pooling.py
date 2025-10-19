import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_undirected

class MincutPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters, dropout=0.5):
        """
        MinCutPool 层实现
        
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度（池化后节点特征维度）
            num_clusters: 聚类数量（池化后节点数量）
            dropout: Dropout 率
        """
        super(MincutPool, self).__init__()
        self.num_clusters = num_clusters
        self.dropout = dropout
        
        # GCN 层用于学习节点嵌入
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        
        # 聚类分配网络
        self.assignment_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, num_clusters)
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch=None, return_loss_components=False):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes]
            return_loss_components: 是否返回损失组件
            
        Returns:
            pooled_x: 池化后的节点特征 [batch_size * num_clusters, out_channels]
            pooled_edge_index: 池化后的边索引 [2, num_pooled_edges]
            pooled_batch: 池化后的批次索引 [batch_size * num_clusters]
            (可选) loss_components: 损失组件字典
        """
        # 处理批次信息
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        batch_size = batch.max().item() + 1
        
        # 1. 通过 GCN 学习节点嵌入
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        
        # 2. 计算聚类分配矩阵 S
        s = self.assignment_net(x)
        s = F.softmax(s, dim=-1)  # 每行和为1
        
        # 3. 转换为密集格式进行计算
        # 转换为密集的邻接矩阵和分配矩阵
        adj_dense = to_dense_adj(edge_index, batch=batch)
        x_dense, mask = to_dense_batch(x, batch=batch)
        s_dense, _ = to_dense_batch(s, batch=batch)
        
        # 4. 使用 dense_mincut_pool 进行池化
        pooled_x, pooled_adj, mincut_loss, ortho_loss = dense_mincut_pool(
            x_dense, adj_dense, s_dense, mask
        )
        
        # 5. 从密集的池化邻接矩阵恢复边索引
        num_pooled_nodes = self.num_clusters
        
        # 创建池化后的边索引和批次信息
        pooled_edge_indices = []
        for i in range(batch_size):
            # 获取当前图的邻接矩阵
            adj_i = pooled_adj[i]
            # 找到非零元素（边）
            edge_index_i = torch.nonzero(adj_i > 0, as_tuple=False).t()
            if edge_index_i.numel() > 0:
                # 添加批次偏移量
                offset = i * num_pooled_nodes
                edge_index_i = edge_index_i + offset
                pooled_edge_indices.append(edge_index_i)
        
        if pooled_edge_indices:
            pooled_edge_index = torch.cat(pooled_edge_indices, dim=1)
            # 确保边索引是整数类型
            pooled_edge_index = pooled_edge_index.long()
            # 转换为无向图（如果需要）
            pooled_edge_index = to_undirected(pooled_edge_index)
        else:
            # 如果没有边，创建空边索引
            pooled_edge_index = torch.zeros(2, 0, dtype=torch.long, device=x.device)
        
        # 6. 创建池化后的批次信息
        # 为每个图的聚类节点分配对应的批次ID
        pooled_batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_pooled_nodes)
        
        # 7. 处理池化后的节点特征
        # 重塑为 [batch_size * num_clusters, hidden_channels]
        pooled_x = pooled_x.reshape(-1, pooled_x.size(-1))
        # 投影到输出维度
        pooled_x = self.output_proj(pooled_x)
        pooled_x = F.relu(pooled_x)
        
        # 8. 准备返回值
        if return_loss_components:
            loss_components = {
                'mincut_loss': mincut_loss,
                'ortho_loss': ortho_loss
            }
            return pooled_x, pooled_edge_index, pooled_batch, loss_components
        
        return pooled_x, pooled_edge_index, pooled_batch
    
    def get_assignment_matrix(self, x, edge_index, batch=None):
        """
        获取聚类分配矩阵（用于可视化或分析）
        """
        with torch.no_grad():
            # 通过 GCN 学习节点嵌入
            x_embed = F.relu(self.gcn1(x, edge_index))
            x_embed = self.gcn2(x_embed, edge_index)
            
            # 计算聚类分配矩阵 S
            s = self.assignment_net(x_embed)
            s = F.softmax(s, dim=-1)
            
            if batch is not None:
                s_dense, _ = to_dense_batch(s, batch=batch)
                return s_dense
            return s