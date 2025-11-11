import torch
from torch import Tensor
from typing import Optional

def compute_mean_average_distance(x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    """
    严格按照原始 GNN 研究中的定义，计算基于余弦距离的均值平均距离 (MAD)。
    MAD = (1 / N(N-1)) * Sum_{i != j} (1 - CosineSimilarity(h_i, h_j))

    Args:
        x (Tensor): 节点的特征矩阵，形状 [N_total, d]。
        batch (Tensor, optional): 节点对应的图索引。如果提供，将计算每个图的 MAD 
                                  并返回平均值。如果未提供，则视为单个图。
                                  
    Returns:
        Tensor: MAD 值（所有节点对余弦距离的平均值）。
    """
    N_total = x.size(0)

    if N_total <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # 1. 如果没有批量索引，视为单个图
    if batch is None:
        return _calculate_single_graph_mad_cosine(x)
        
    # 2. 如果提供了批量索引，对每个图分别计算 MAD
    else:
        unique_graphs = batch.unique()
        mad_list = []
        
        for graph_idx in unique_graphs:
            # 筛选出当前图的节点特征
            x_g = x[batch == graph_idx]
            N_g = x_g.size(0)
            
            if N_g > 1:
                mad_g = _calculate_single_graph_mad_cosine(x_g)
                mad_list.append(mad_g)

        if not mad_list:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
        # 返回所有图的 MAD 的平均值
        return torch.stack(mad_list).mean()


def _calculate_single_graph_mad_cosine(x: Tensor) -> Tensor:
    """
    辅助函数：计算单个图的 MAD（基于余弦距离）。
    """
    N = x.size(0)
    
    # 1. 归一化特征向量，使其范数为 1
    # 增加一个小 epsilon 防止除以零
    norm_x = torch.linalg.norm(x, dim=1, keepdim=True)
    x_norm = x / (norm_x + 1e-8) 

    # 2. 计算余弦相似度矩阵 (Cosine Similarity)
    # CosineSim_{ij} = h_i @ h_j.T，形状: [N, N]
    cosine_sim_matrix = torch.matmul(x_norm, x_norm.T) 
    
    # 3. 转换为余弦距离 (Distance = 1 - Similarity)
    # CosineDist_{ij} = 1 - CosineSim_{ij}
    cosine_dist_matrix = 1.0 - cosine_sim_matrix
    
    # 4. 累加所有距离并求平均 (排除 i=j 的对角线元素)
    
    # 对角线元素 CosineDist_{ii} 理论上为 0 (1 - 1 = 0)，但可能因浮点误差不为 0
    # 为了严谨，确保只对 i != j 的距离进行求和
    
    # 排除对角线元素的一种安全方法是先求和，再减去对角线元素
    total_distance = torch.sum(cosine_dist_matrix) - torch.sum(torch.diag(cosine_dist_matrix))
    
    # 节点对总数是 N * (N - 1)
    mad = total_distance / (N * (N - 1))
    
    return mad