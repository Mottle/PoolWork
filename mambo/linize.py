import torch
from torch import max, mean, sum
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

def convert_to_line_graph_with_batch_vectorized_clean(x, edge_index, batch, merge_fn = torch.max, logging_enable = False):
    """
    将点图转换为边图（线图），支持batch处理，使用向量化操作提高效率，去除自环和重边
    
    Args:
        data: PyG Data对象，包含x、edge_index和batch
        
    Returns:
        line_data: 边图的PyG Data对象，包含x_line、edge_index_line和batch_line
    """
    
    num_edges_original = edge_index.size(1)
    num_nodes = x.size(0)
    
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
    
    num_graphs = batch.max().item() + 1
    
    if logging_enable:
        print(f"原图: {num_nodes}个节点, {num_edges_original}条边, {num_graphs}个图")
    
    # 步骤1: 去除自环边
    edge_index_no_self, _ = remove_self_loops(edge_index)
    num_edges_no_self = edge_index_no_self.size(1)
    
    if logging_enable:
        print(f"去除自环后: {num_edges_no_self}条边")
    
    # 步骤2: 去除重边并合并
    edge_index_clean, _ = coalesce(edge_index_no_self, None, num_nodes, num_nodes)
    num_edges_clean = edge_index_clean.size(1)
    if logging_enable:
        print(f"去除重边后: {num_edges_clean}条边")
    
    # 步骤3: 构建边图的节点特征和batch
    src_nodes = edge_index_clean[0]
    dst_nodes = edge_index_clean[1]
    # x_line = torch.cat([x[src_nodes], x[dst_nodes]], dim=1)
    x_line = merge_fn(x[src_nodes], x[dst_nodes])
    batch_line = batch[src_nodes]
    
    if logging_enable:
        print(f"边图节点特征形状: {x_line.shape}")
    
    # 步骤4: 构建边图的边索引（向量化版本）
    # 创建一个从节点到边的映射（使用稀疏矩阵）
    row, col = edge_index_clean
    node_to_edge_map = torch.zeros((num_nodes, num_edges_clean), device=x.device)
    
    # 使用索引赋值，确保映射正确
    edge_indices = torch.arange(num_edges_clean, device=x.device)
    node_to_edge_map[row, edge_indices] = 1
    node_to_edge_map[col, edge_indices] = 1
    
    # 对于每个节点，找到连接它的所有边
    line_edges = []
    
    for graph_idx in range(num_graphs):
        # 获取当前图的节点
        graph_nodes = torch.where(batch == graph_idx)[0]
        
        for node in graph_nodes:
            # 找到连接到当前节点的所有边
            connected_edges = torch.where(node_to_edge_map[node])[0]
            
            if len(connected_edges) > 1:
                # 创建这些边之间的完全连接
                for i in range(len(connected_edges)):
                    for j in range(i + 1, len(connected_edges)):
                        line_edges.append([connected_edges[i].item(), connected_edges[j].item()])
    
    if line_edges:
        edge_index_line = torch.tensor(line_edges, dtype=torch.long).t().contiguous()
        edge_index_line = to_undirected(edge_index_line)
    else:
        edge_index_line = torch.empty((2, 0), dtype=torch.long)
    
    if logging_enable:
        print(f"边图: {x_line.size(0)}个节点, {edge_index_line.size(1)}条边")
    
    # 创建边图的Data对象
    line_data = Data(x=x_line.to(x.device), edge_index=edge_index_line.to(x.device), batch=batch_line.to(x.device))
    
    # 保存原始信息用于验证
    line_data.original_edge_index = edge_index_clean.clone()
    line_data.removed_self_loops = num_edges_original - num_edges_no_self
    line_data.removed_duplicates = num_edges_no_self - num_edges_clean
    
    return line_data