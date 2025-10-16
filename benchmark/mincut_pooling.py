import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch, degree
from torch_geometric.nn.inits import glorot, zeros
import numpy as np


class MinCutPooling(nn.Module):
    """
    MinCutPool: 基于谱聚类和图割最小化的图池化层
    
    参考文献: 
    "Spectral Clustering with Graph Neural Networks for Graph Pooling"
    """
    
    def __init__(self, in_channels, k, num_mlps=1, dropout=0.1, 
                 lambd=1.0, ortho_penalty=True, **kwargs):
        """
        初始化 MinCutPool
        
        Args:
            in_channels (int): 输入特征维度
            k (int): 池化后的簇数量（节点数）
            num_mlps (int): MLP层数，用于计算分配矩阵
            dropout (float): Dropout比率
            lambd (float): 正交性惩罚项的权重
            ortho_penalty (bool): 是否使用正交性惩罚
        """
        super(MinCutPooling, self).__init__()
        
        self.in_channels = in_channels
        self.k = k
        self.lambd = lambd
        self.ortho_penalty = ortho_penalty
        
        # MLP用于计算软分配矩阵
        self.mlp_layers = nn.ModuleList()
        
        # 第一层
        self.mlp_layers.append(nn.Linear(in_channels, in_channels))
        
        # 中间层
        for _ in range(num_mlps - 1):
            self.mlp_layers.append(nn.Linear(in_channels, in_channels))
        
        # 输出层 - 输出k维，对应k个簇
        self.mlp_out = nn.Linear(in_channels, k)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        for layer in self.mlp_layers:
            glorot(layer.weight)
            zeros(layer.bias)
        glorot(self.mlp_out.weight)
        zeros(self.mlp_out.bias)
    
    def forward(self, x, edge_index, batch=None, return_assignment=False):
        """
        前向传播
        
        Args:
            x (Tensor): 节点特征 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            batch (Tensor, optional): 批次向量 [N]
            return_assignment (bool): 是否返回分配矩阵
            
        Returns:
            x_pooled (Tensor): 池化后节点特征 [B * k, in_channels]
            adj_pooled (Tensor): 池化后邻接矩阵 [B * k, B * k]
            batch_pooled (Tensor): 池化后批次向量 [B * k]
            mc_loss (Tensor): MinCut损失项
            ortho_loss (Tensor): 正交性损失项
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 1. 计算软分配矩阵 S
        S = self._compute_assignment_matrix(x)
        
        # 2. 将图数据转换为批次密集表示
        x_dense, mask = to_dense_batch(x, batch)  # [B, N_max, in_channels]
        adj_dense = to_dense_adj(edge_index, batch)  # [B, N_max, N_max]
        
        batch_size = x_dense.size(0)
        num_nodes = x_dense.size(1)
        
        # 3. 将分配矩阵转换为密集形式
        S_dense, _ = to_dense_batch(S, batch)  # [B, N_max, k]
        
        # 4. 计算池化后的特征和邻接矩阵
        x_pooled = self._pool_features(x_dense, S_dense, mask)  # [B, k, in_channels]
        adj_pooled = self._pool_adjacency(adj_dense, S_dense, mask)  # [B, k, k]
        
        # 5. 展平批次维度
        x_pooled = x_pooled.reshape(batch_size * self.k, -1)  # [B * k, in_channels]
        adj_pooled = self._flatten_adjacency(adj_pooled, batch_size)  # [B * k, B * k]
        batch_pooled = torch.repeat_interleave(
            torch.arange(batch_size, device=x.device), self.k
        )  # [B * k]
        
        # 6. 计算MinCut损失和正交性损失
        mc_loss, ortho_loss = self._compute_loss(adj_dense, S_dense, mask)
        
        if return_assignment:
            return x_pooled, adj_pooled, batch_pooled, mc_loss, ortho_loss, S
        
        return x_pooled, adj_pooled, batch_pooled, mc_loss, ortho_loss
    
    def _compute_assignment_matrix(self, x):
        """计算软分配矩阵 S [N, k]"""
        # 通过MLP计算分配分数
        h = x
        for layer in self.mlp_layers:
            h = self.activation(layer(h))
            h = self.dropout(h)
        
        # 输出层 - 不使用激活函数
        S = self.mlp_out(h)  # [N, k]
        
        # 应用softmax得到概率分布
        S = F.softmax(S, dim=-1)
        
        return S
    
    def _pool_features(self, x_dense, S_dense, mask):
        """
        池化节点特征
        
        Args:
            x_dense: [B, N_max, in_channels]
            S_dense: [B, N_max, k] 
            mask: [B, N_max]
            
        Returns:
            x_pooled: [B, k, in_channels]
        """
        # 掩码处理 - 将填充位置的分配概率设为0
        S_masked = S_dense * mask.unsqueeze(-1).float()
        
        # 归一化：每个簇的节点数
        cluster_size = S_masked.sum(1)  # [B, k]
        cluster_size = torch.clamp(cluster_size, min=1.0)  # 避免除零
        
        # 池化特征: X_pooled = S^T * X / cluster_size
        x_pooled = torch.bmm(S_masked.transpose(1, 2), x_dense)  # [B, k, in_channels]
        x_pooled = x_pooled / cluster_size.unsqueeze(-1)
        
        return x_pooled
    
    def _pool_adjacency(self, adj_dense, S_dense, mask):
        """
        池化邻接矩阵
        
        Args:
            adj_dense: [B, N_max, N_max]
            S_dense: [B, N_max, k]
            mask: [B, N_max]
            
        Returns:
            adj_pooled: [B, k, k]
        """
        # 掩码处理
        S_masked = S_dense * mask.unsqueeze(-1).float()
        
        # 池化邻接矩阵: A_pooled = S^T * A * S
        adj_pooled = torch.bmm(S_masked.transpose(1, 2), torch.bmm(adj_dense, S_masked))
        
        return adj_pooled
    
    def _flatten_adjacency(self, adj_pooled, batch_size):
        """
        将批次邻接矩阵展平为单个大图
        
        Args:
            adj_pooled: [B, k, k]
            batch_size: int
            
        Returns:
            adj_flat: [B * k, B * k]
        """
        # 创建块对角矩阵
        adj_flat = torch.zeros(
            batch_size * self.k, batch_size * self.k,
            device=adj_pooled.device
        )
        
        for i in range(batch_size):
            start_idx = i * self.k
            end_idx = (i + 1) * self.k
            adj_flat[start_idx:end_idx, start_idx:end_idx] = adj_pooled[i]
        
        return adj_flat
    
    def _compute_loss(self, adj_dense, S_dense, mask):
        """
        计算MinCut损失和正交性损失
        
        Args:
            adj_dense: [B, N_max, N_max]
            S_dense: [B, N_max, k]
            mask: [B, N_max]
            
        Returns:
            mc_loss: MinCut损失
            ortho_loss: 正交性损失
        """
        batch_size = adj_dense.size(0)
        
        mc_loss = 0.0
        ortho_loss = 0.0
        
        for i in range(batch_size):
            # 提取当前图的邻接矩阵和分配矩阵
            A = adj_dense[i]  # [N_max, N_max]
            S = S_dense[i]  # [N_max, k]
            m = mask[i]  # [N_max]
            
            # 掩码处理
            num_nodes = int(m.sum())
            A = A[:num_nodes, :num_nodes]  # [num_nodes, num_nodes]
            S = S[:num_nodes]  # [num_nodes, k]
            
            # 计算度矩阵
            deg = A.sum(dim=1)  # [num_nodes]
            D = torch.diag(deg)  # [num_nodes, num_nodes]
            
            # MinCut损失: -Tr(S^T A S) / Tr(S^T D S)
            numerator = -torch.trace(S.t() @ A @ S)
            denominator = torch.trace(S.t() @ D @ S)
            
            # 避免数值不稳定
            denominator = torch.clamp(denominator, min=1e-8)
            
            graph_mc_loss = numerator / denominator
            mc_loss += graph_mc_loss
            
            # 正交性损失: ||S^T S / ||S^T S||_F - I/sqrt(k)||_F
            if self.ortho_penalty:
                S_t_S = S.t() @ S  # [k, k]
                norm_S_t_S = torch.norm(S_t_S, p='fro')
                
                # 避免除零
                norm_S_t_S = torch.clamp(norm_S_t_S, min=1e-8)
                
                normalized_S_t_S = S_t_S / norm_S_t_S
                I_over_sqrt_k = torch.eye(self.k, device=S.device) / np.sqrt(self.k)
                
                graph_ortho_loss = torch.norm(normalized_S_t_S - I_over_sqrt_k, p='fro')
                ortho_loss += graph_ortho_loss
        
        # 平均损失
        mc_loss = mc_loss / batch_size
        ortho_loss = ortho_loss / batch_size
        
        return mc_loss, ortho_loss
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, k={self.k})'


class MinCutPoolingWithGNN(nn.Module):
    """
    使用GNN计算分配矩阵的MinCutPool变体
    """
    
    def __init__(self, in_channels, k, gnn_hidden=64, gnn_layers=2, 
                 dropout=0.1, lambd=1.0, **kwargs):
        super(MinCutPoolingWithGNN, self).__init__()
        
        self.in_channels = in_channels
        self.k = k
        self.lambd = lambd
        
        # GNN层用于计算分配矩阵
        self.gnn_layers = nn.ModuleList()
        
        # 第一层
        self.gnn_layers.append(GCNConv(in_channels, gnn_hidden))
        
        # 中间层
        for _ in range(gnn_layers - 2):
            self.gnn_layers.append(GCNConv(gnn_hidden, gnn_hidden))
        
        # 最后一层 - 输出k维分配
        if gnn_layers > 1:
            self.gnn_layers.append(GCNConv(gnn_hidden, k))
        else:
            self.gnn_layers.append(GCNConv(in_channels, k))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # 基础MinCutPool（重用池化逻辑）
        self.mincut_pool = MinCutPooling(in_channels, k, lambd=lambd, **kwargs)
    
    def _compute_assignment_matrix(self, x, edge_index):
        """使用GNN计算分配矩阵"""
        h = x
        
        # 应用GNN层
        for i, layer in enumerate(self.gnn_layers):
            h = layer(h, edge_index)
            if i < len(self.gnn_layers) - 1:  # 除了最后一层外都使用激活和dropout
                h = self.activation(h)
                h = self.dropout(h)
        
        # 应用softmax得到概率分布
        S = F.softmax(h, dim=-1)
        
        return S
    
    def forward(self, x, edge_index, batch=None, return_assignment=False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 使用GNN计算分配矩阵
        S = self._compute_assignment_matrix(x, edge_index)
        
        # 将图数据转换为批次密集表示
        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch)
        S_dense, _ = to_dense_batch(S, batch)
        
        batch_size = x_dense.size(0)
        
        # 池化特征和邻接矩阵
        x_pooled = self.mincut_pool._pool_features(x_dense, S_dense, mask)
        adj_pooled = self.mincut_pool._pool_adjacency(adj_dense, S_dense, mask)
        
        # 展平批次维度
        x_pooled = x_pooled.reshape(batch_size * self.k, -1)
        adj_pooled = self.mincut_pool._flatten_adjacency(adj_pooled, batch_size)
        batch_pooled = torch.repeat_interleave(
            torch.arange(batch_size, device=x.device), self.k
        )
        
        # 计算损失
        mc_loss, ortho_loss = self.mincut_pool._compute_loss(adj_dense, S_dense, mask)
        
        if return_assignment:
            return x_pooled, adj_pooled, batch_pooled, mc_loss, ortho_loss, S
        
        return x_pooled, adj_pooled, batch_pooled, mc_loss, ortho_loss
