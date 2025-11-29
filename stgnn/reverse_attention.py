import torch
from torch.nn import functional as F
from torch_scatter import scatter_max, scatter_sum, scatter_mean
from torch_geometric.utils import softmax
from typing import Tuple, Optional

# def reverse_self_attention(
#     x: torch.Tensor, 
#     batch: torch.Tensor
# ) -> torch.Tensor:
#     """
#     实现 PyG Batch 感知的图内 Reverse Global Self-Attention。
#     确保注意力计算只发生在同一图 (同一 batch ID) 的节点之间。

#     Args:
#         x: 节点特征矩阵 (N, D)。N 是所有图中的节点总数。
#         batch: 节点到图索引的映射 (N,)。每个元素是其对应节点所属图的 Batch ID (0, 1, 2, ...)。

#     Returns:
#         x_out: 加权聚合后的新特征矩阵 (N, D)。
#         attention_matrix: 稀疏化的注意力权重矩阵 (N, N) -- 注意：这里无法返回稠密矩阵。
#                          我们返回用于计算的 scores 和 batch 索引以供检查。
#     """
#     N, D = x.shape
    
#     # 1. 计算所有节点两两之间的点积相似度 S (N, N)
#     # S[i, j] 是节点 i 和节点 j 的点积
#     # 注意：这里计算了跨图的相似度，但后面会通过 Masking 和 Group Softmax 忽略它们。
#     similarity_matrix = torch.matmul(x, x.transpose(-1, -2)) # (N, N)

#     # 2. 计算 Reverse Score 矩阵 R (N, N)
#     # R[i, j] 是节点 i 关注节点 j 的原始得分。
#     reverse_score_matrix = -similarity_matrix

#     # --- 关键步骤：分组 Softmax ---
    
#     # 3. 构造 Group Mask (可选，但推荐)
#     # 创建一个张量，指示节点 i 和 j 是否属于同一个 Batch ID (图)。
#     # batch: (N,); batch.unsqueeze(1): (N, 1)
#     # mask: (N, N). mask[i, j] == True if batch[i] == batch[j]
#     mask = (batch.unsqueeze(1) == batch.unsqueeze(0)) 
    
#     # 4. 应用 Masking 和分组 Softmax

#     # A. 屏蔽跨图的得分
#     # 将跨图的得分设置为一个极小值，以确保其 Softmax 权重趋近于 0。
#     # 如果节点 i 和 j 不在同一图，mask[i, j] == False
#     neg_inf = torch.finfo(reverse_score_matrix.dtype).min
#     masked_scores = reverse_score_matrix.masked_fill(~mask, neg_inf)
    
#     # B. 分组 Softmax (手工实现)
    
#     # 4.1. 计算每个 Query 节点 i 所在组的最大得分 (Max-Subtracted Trick for stability)
#     # 我们需要找到每个图/Batch ID 内部，所有 Query 行的最大值。
#     # masked_scores.max(dim=-1)[0] 得到每行的最大值 (N,)
#     # scatter_max(..., index=batch) 得到每个 Batch ID 的最大值
#     max_scores, _ = scatter_max(masked_scores.max(dim=-1)[0], batch) # (B,)
#     # 扩展回 (N,)
#     max_scores_per_node = max_scores[batch] 
    
#     # 4.2. Max-Subtraction (稳定化)
#     # 减去图内的最大值，防止溢出
#     # masked_scores - max_scores_per_node.unsqueeze(1)
#     score_stable = masked_scores - max_scores_per_node.unsqueeze(1) 
    
#     # 4.3. Exponentiation
#     exp_score = torch.exp(score_stable)
    
#     # 4.4. 分组 Sum (Softmax 分母)
#     # exp_score.sum(dim=-1) 得到每行的指数和 (N,)
#     # scatter_sum(..., index=batch) 得到每个 Batch ID 的总和
#     sum_exp_score, _ = scatter_sum(exp_score.sum(dim=-1), batch) # (B,)
#     # 扩展回 (N,)
#     sum_exp_score_per_node = sum_exp_score[batch]

#     # 4.5. 最终 Attention Weights A (N, N)
#     attention_matrix = exp_score / sum_exp_score_per_node.unsqueeze(1)

#     # --- 聚合 ---
    
#     # 5. 加权求和得到输出 X' (N, D)
#     # A @ X : (N, N) @ (N, D) -> (N, D)
#     x_out = torch.matmul(attention_matrix, x)

#     # 注意：由于 Softmax 分母是手工计算的，需要确保它不为零。
#     # 在 Softmax(dim=-1) 中，如果所有值都是 neg_inf，PyTorch 会自动处理。
    
#     return x_out, reverse_score_matrix, attention_matrix # 返回 R 和 A 用于检查

def reverse_self_attention(
    x: torch.Tensor,
    batch: torch.Tensor
) -> torch.Tensor:
    """
    高效实现 PyG Batch 感知的图内 Reverse Global Self-Attention。
    使用稀疏边索引避免构造 N×N 矩阵，保持输出顺序。

    Args:
        x: 节点特征矩阵 (N, D)
        batch: 节点到图索引的映射 (N,)

    Returns:
        x_out: 加权聚合后的新特征矩阵 (N, D)
        reverse_scores: 边上的原始得分 (E,)
        attention_weights: 边上的注意力权重 (E,)
        edge_index: 边索引 (2, E)
    """
    N, D = x.size()
    device = x.device

    # 构造图内所有 (i, j) 边索引
    edge_index_i = []
    edge_index_j = []

    for b in batch.unique():
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        row, col = torch.meshgrid(idx, idx, indexing='ij')
        edge_index_i.append(row.flatten())
        edge_index_j.append(col.flatten())

    row_idx = torch.cat(edge_index_i, dim=0)
    col_idx = torch.cat(edge_index_j, dim=0)
    # edge_index = torch.stack([row_idx, col_idx], dim=0)  # (2, E)

    # 计算 Reverse Score：负的点积
    reverse_scores = -(x[row_idx] * x[col_idx]).sum(dim=-1)  # (E,)

    # 分组 Softmax：按 query 节点 row_idx 分组
    attention_weights = softmax(reverse_scores, row_idx)

    # 加权聚合：按 query 节点 row_idx 聚合 value 节点 col_idx 的特征
    x_out = torch.zeros_like(x)
    x_out.index_add_(0, row_idx, attention_weights.unsqueeze(-1) * x[col_idx])

    return x_out, reverse_scores, attention_weights

def centered_tanh_attention_weights(
    pre_activation_scores: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: Optional[int] = None
) -> torch.Tensor:
    """
    使用 tanh 生成 [-1, 1] 范围的得分，然后对每个组进行中心化，
    使得每个组内的权重平均值为 0。
    
    Args:
        pre_activation_scores: 原始的注意力得分，形状为 [num_edges]。
        group_index: 分组索引 (例如，边的目标节点索引)，形状为 [num_edges]。
        num_groups: 分组的总数 (例如，图中的节点总数)。
        
    Returns:
        中心化后的注意力权重，每个组的总和趋于 0。
    """
    # 1. 应用 tanh 函数，范围在 [-1, 1]
    tanh_scores = torch.tanh(pre_activation_scores)
    
    # 2. 计算每个分组的平均值 (Mean Aggregation)
    # scatter_mean(src, index, dim=0, dim_size=None)
    # 计算具有相同 group_index 的元素的平均值。
    group_mean = scatter_mean(
        src=tanh_scores, 
        index=group_index, 
        dim=0, 
        dim_size=num_groups
    )
    
    # 3. 对每个元素减去其所在分组的平均值 (中心化)
    # 使用 group_index 来广播 group_mean 到对应的边上。
    # group_mean[group_index] 会将计算出的平均值对齐到原始的边张量。
    centered_scores = tanh_scores - group_mean[group_index]
    
    return centered_scores

def bidirectional_reverse_self_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    batch: torch.Tensor
) -> torch.Tensor:
    # 构造图内所有 (i, j) 边索引
    edge_index_i = []
    edge_index_j = []

    for b in batch.unique():
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        row, col = torch.meshgrid(idx, idx, indexing='ij')
        edge_index_i.append(row.flatten())
        edge_index_j.append(col.flatten())

    row_idx = torch.cat(edge_index_i, dim=0)
    col_idx = torch.cat(edge_index_j, dim=0)
    # edge_index = torch.stack([row_idx, col_idx], dim=0)  # (2, E)

    # 计算 Reverse Score：负的点积
    reverse_scores = -(Q[row_idx] * K[col_idx]).sum(dim=-1)  # (E,)

    # 分组 Softmax：按 query 节点 row_idx 分组
    # attention_weights = softmax(reverse_scores, row_idx)
    attention_weights = centered_tanh_attention_weights(reverse_scores, row_idx)

    # 加权聚合：按 query 节点 row_idx 聚合 value 节点 col_idx 的特征
    x_out = torch.zeros_like(V)
    x_out.index_add_(0, row_idx, attention_weights.unsqueeze(-1) * V[col_idx])

    return x_out, reverse_scores, attention_weights

# class DynamicReverseAttention(torch.nn.Module):