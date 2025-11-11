import torch
from torch import Tensor

def compute_dirichlet_energy(x: Tensor, edge_index: Tensor, batch: Tensor = None) -> Tensor:
    """
    计算 PyG Batch 对象中所有图的狄利克雷能量之和。
    
    E(H) = 0.5 * Sum_{i, j in Edges} || h_i - h_j ||^2
    
    Args:
        x (Tensor): 所有图的节点特征，形状 [N_total, d]。
        edge_index (Tensor): 调整后的边索引，形状 [2, E_total]。
        batch (Tensor, optional): 节点对应的图索引，形状 [N_total]。
                                  虽然在计算中未使用，但保留以便接口一致性。
                                  
    Returns:
        Tensor: 批次中所有图的狄利克雷能量总和。
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # 1. 获取相邻节点的特征
    # edge_index[0] 包含源节点索引 (i)
    # edge_index[1] 包含目标节点索引 (j)
    h_i = x[edge_index[0]] # 形状 [E_total, d]
    h_j = x[edge_index[1]] # 形状 [E_total, d]

    # 2. 计算相邻节点特征的 L2 距离平方
    # (h_i - h_j)^2
    diff_sq = (h_i - h_j)**2 
    
    # 3. 对特征维度求和，得到每条边的 L2 距离平方
    # || h_i - h_j ||^2
    l2_dist_sq = torch.sum(diff_sq, dim=1) # 形状 [E_total]

    # 4. 对所有边求和
    # PyG 的 edge_index 通常包含双向边 (i, j) 和 (j, i)，
    # 公式 E(H) = 0.5 * Sum_{i, j} A_ij || h_i - h_j ||^2
    dirichlet_energy = torch.sum(l2_dist_sq)
    
    return dirichlet_energy