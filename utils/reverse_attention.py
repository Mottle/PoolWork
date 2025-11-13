import torch
from torch.nn import functional as F

def reverse_self_attention(
    x: torch.Tensor, 
    batch_size: int
) -> torch.Tensor:
    """
    实现基于原始点积的全局 Reverse Self-Attention，通过参数引入 Batch 维度。
    
    Args:
        x: 节点特征矩阵 (N, D)。N: 节点数, D: 特征维度。
        batch_size: 期望模拟的 Batch Size (B)。
    
    Returns:
        x_out: 加权聚合后的新特征矩阵 (B, N, D)。
        attention_matrix: 注意力权重矩阵 (B, N, N)。
    """
    
    N, D = x.shape
    B = batch_size

    # 1. 扩展 X 以引入 Batch 维度: (N, D) -> (1, N, D) -> (B, N, D)
    # 这一步是关键，将 (N, D) 的特征复制 B 份。
    x_batched = x.unsqueeze(0).repeat(B, 1, 1) # (B, N, D)

    # --- 后续计算与 Batch-aware 版本一致 ---

    # 2. 计算点积相似度矩阵 S: (B, N, N)
    # S = X @ X_T
    similarity_matrix = torch.matmul(x_batched, x_batched.transpose(-1, -2)) 

    # 3. 计算 Reverse Score 矩阵 R: (B, N, N)
    reverse_score_matrix = -similarity_matrix

    # 4. Attention 矩阵 A: 对最后一维 (N) 进行 Softmax
    attention_matrix = F.softmax(reverse_score_matrix, dim=-1) # (B, N, N)

    # 5. 加权求和得到输出 X': (B, N, D)
    # X' = A @ X
    x_out = torch.matmul(attention_matrix, x_batched)

    return x_out, attention_matrix