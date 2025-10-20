import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import remove_self_loops, softmax, coalesce, scatter
from typing import Callable, List, NamedTuple, Optional, Tuple
from torch import Tensor
from perf_counter import measure_time

#先使用GNN学习分配矩阵 S: V -> V', 分配出V'个聚类中心，
#再使用GAT计算出边权重得分，随后应用edge pooling方法完成聚类
#在聚类特征计算部分，使用注意力学习每个节点对聚类中心节点的贡献度
class MamboPoolingWithClusterAttention(nn.Module):
    def __init__(self, input_dim: int, reduce_ratio: float = 0.2, cluster_center_heads: int = 4, edge_score_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.reduce_ratio = reduce_ratio
        self.cluster_center_heads = cluster_center_heads
        self.edge_score_heads = edge_score_heads

        if reduce_ratio > 1 or reduce_ratio <= 0:
            raise ValueError("Ratio must be between 0 and 1")
        
        self._build_cluster_center_attention()
        self._build_edge_score_gnn()

    #节点对全局特征的注意力，用于后续TopK选择分配中心
    def _build_cluster_center_attention(self):
        self.cluster_center_attention = MultiheadAttention(embed_dim=self.input_dim, num_heads=self.cluster_center_heads, dropout=0.2)

    def _build_edge_score_gnn(self):
        self.edge_score_conv = GATConv(self.input_dim, self.input_dim, heads=self.edge_score_heads, dropout=0.2)
        self.attention_reduce = torch.nn.Linear(self.edge_score_heads, 1)

    def forward(self, x, edge_index, batch):
        # compute_cluster_center_attention = measure_time(self._compute_cluster_center_attention_optimized)
        # select_topk = measure_time(self._select_topk_optimized)
        # compute_edge_scores = measure_time(self._compute_edge_scores)
        # pool = measure_time(self._pool_parallel_more)

        # (cluster_center_attn, global_feature_attned), t_center_att = compute_cluster_center_attention(x, batch)

        # (topk_indices, topk_batch), t_topk = select_topk(cluster_center_attn, batch)
        
        # (x_attned, clean_edge_index, clean_edge_score), t_edge_s = compute_edge_scores(x, edge_index)
        
        # x = x + x_attned

        # (pooled_x, pooled_edge_index, _, pooled_batch), t_pool = pool(
        #     x=x,
        #     edge_index=clean_edge_index,
        #     batch=batch, 
        #     global_feature=global_feature_attned, 
        #     cluster_center_indices=topk_indices, 
        #     cluster_center_batch=topk_batch, 
        #     edge_score=clean_edge_score
        # )
        # all_time = t_center_att + t_topk + t_edge_s + t_pool
        # print(f"All time: {all_time}, Cluster center attention time: {t_center_att/all_time:.4f}:.3f, TopK time: {t_topk/all_time:.4f}, Edge score time: {t_edge_s/all_time:.4f}, Pool time: {t_pool/all_time:.4f}")


        cluster_center_attn, global_feature_attned = self._compute_cluster_center_attention_optimized(x, batch)

        topk_indices, topk_batch = self._select_topk_optimized(cluster_center_attn, batch)
        
        x_attned, clean_edge_index, clean_edge_score = self._compute_edge_scores(x, edge_index)
        
        x = x + x_attned

        pooled_x, pooled_edge_index, _, pooled_batch = self._pool_parallel_more(
            x=x,
            edge_index=clean_edge_index,
            batch=batch, 
            global_feature=global_feature_attned, 
            cluster_center_indices=topk_indices, 
            cluster_center_batch=topk_batch, 
            edge_score=clean_edge_score
        )

        return pooled_x, pooled_edge_index, pooled_batch
    

    # def _compute_cluster_center_attention(self, x, batch):
    #     graph_feature = global_mean_pool(x=x, batch=batch)
    #     batch_size = torch.unique(batch).size(0)
    #     att = []
    #     for batch_index in range(batch_size):
    #         mask = batch == batch_index
    #         batch_x = x[mask]
    #         batch_graph_feature = graph_feature[batch_index]
    #         cluster_center_attention, _ = self.cluster_center_attention(batch_x, batch_graph_feature, batch_graph_feature)
    #         att.append(cluster_center_attention)
    #     cluster_center_attn = torch.cat(att, dim=0)
    #     graph_feature_attn = graph_feature[batch] + x * cluster_center_attn.unsqueeze(1)

    #     return cluster_center_attention, graph_feature_attn

    def _compute_cluster_center_attention(self, x, batch):
        graph_feature = global_mean_pool(x=x, batch=batch)
        batch_size = torch.unique(batch).size(0)
        att = []
        for batch_index in range(batch_size):
            mask = batch == batch_index
            batch_x = x[mask]
            batch_graph_feature = graph_feature[batch_index]
            
            # 确保batch_graph_feature是2-D
            batch_graph_feature_2d = batch_graph_feature.unsqueeze(0)  # [1, feature_dim]
            
            # 扩展batch_graph_feature以匹配batch_x的序列长度
            batch_graph_feature_expanded = batch_graph_feature_2d.expand(batch_x.size(0), -1)  # [num_nodes, feature_dim]
            
            # 调用注意力机制
            cluster_center_attention, _ = self.cluster_center_attention(
                batch_x, 
                batch_graph_feature_expanded, 
                batch_graph_feature_expanded
            )
            att.append(cluster_center_attention)
        
        cluster_center_attn = torch.cat(att, dim=0)
        # graph_feature_attn = graph_feature[batch] + x * cluster_center_attn.unsqueeze(1)
        weighted_x = x * cluster_center_attn#.unsqueeze(1)
        new_graph_feature = global_mean_pool(weighted_x, batch) + graph_feature

        return cluster_center_attn, new_graph_feature

    def _compute_cluster_center_attention_optimized(self, x, batch):
        # Step 1: 计算每个图的全局特征
        graph_feature = global_mean_pool(x, batch)  # [num_graphs, feature_dim]

        # Step 2: 将每个节点对应的图特征广播到节点维度
        graph_feature_per_node = graph_feature[batch]  # [num_nodes, feature_dim]

        # Step 3: 一次性调用注意力机制
        cluster_center_attention, _ = self.cluster_center_attention(
            x, graph_feature_per_node, graph_feature_per_node
        )  # [num_nodes]

        # Step 4: 加权节点特征
        weighted_x = x * cluster_center_attention  # [num_nodes, feature_dim]

        # Step 5: 聚合得到新的图特征
        new_graph_feature = global_mean_pool(weighted_x, batch) + graph_feature  # [num_graphs, feature_dim]

        return cluster_center_attention, new_graph_feature


    def _select_topk(self, cluster_center_attn, batch):

        cluster_center_attn = cluster_center_attn#.to('cpu')
        batch = batch#.to('cpu')

        cluster_center_attn = torch.sum(cluster_center_attn, dim=-1)

        # 计算每个批次的元素数量和要选取的topk数量
        count = torch.bincount(batch)
        k = torch.ceil(count * self.reduce_ratio).int()
        
        # 获取所有批次ID
        batch_ids = torch.unique(batch)
        all_topk_indices = []
        all_topk_batch = []

        for i, batch_id in enumerate(batch_ids):
            # 获取当前批次的所有元素
            batch_mask = (batch == batch_id)
            batch_attn = cluster_center_attn[batch_mask]
            
            # 确定当前批次要选取的k值 - 使用枚举索引i而不是batch_id
            current_k = min(k[i], len(batch_attn))
            
            if current_k > 0:
                # 选取topk
                sorted_indices = torch.argsort(batch_attn, descending=True)
                topk_local_indices = sorted_indices[:current_k]
                
                # 转换为全局索引
                batch_indices = torch.where(batch_mask)[0]
                topk_global_indices = batch_indices[topk_local_indices]
                all_topk_indices.append(topk_global_indices)
                
                # 创建对应的批次标签
                topk_batch = torch.full((current_k,), batch_id, dtype=torch.long, device=batch.device)
                all_topk_batch.append(topk_batch)

        # 合并所有批次的结果
        if all_topk_indices:
            topk_indices = torch.cat(all_topk_indices)
            topk_batch = torch.cat(all_topk_batch)
        else:
            topk_indices = torch.tensor([], dtype=torch.long, device=batch.device)
            topk_batch = torch.tensor([], dtype=torch.long, device=batch.device)
        
        return topk_indices, topk_batch

    def _select_topk_optimized(self, cluster_center_attn, batch):
        # 聚合注意力分数
        attn_score = torch.sum(cluster_center_attn, dim=-1)

        # 获取 batch 信息
        batch_size = batch.max().item() + 1
        count = torch.bincount(batch, minlength=batch_size)
        k = torch.ceil(count * self.reduce_ratio).int()

        # 为每个 batch 构建索引列表
        sorted_batch = batch.argsort()
        batch_sorted = batch[sorted_batch]
        attn_sorted = attn_score[sorted_batch]

        # 构建每个 batch 的起始位置
        batch_ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=batch.device)
        batch_ptr[1:] = torch.cumsum(count, dim=0)

        topk_indices = []
        topk_batch = []

        for i in range(batch_size):
            start = batch_ptr[i]
            end = batch_ptr[i + 1]
            if end > start and k[i] > 0:
                current_attn = attn_sorted[start:end]
                current_topk = min(k[i], end - start)
                _, local_topk = torch.topk(current_attn, current_topk, largest=True, sorted=False)
                global_topk = sorted_batch[start + local_topk]
                topk_indices.append(global_topk)
                topk_batch.append(torch.full((current_topk,), i, dtype=torch.long, device=batch.device))

        if topk_indices:
            topk_indices = torch.cat(topk_indices)
            topk_batch = torch.cat(topk_batch)
        else:
            topk_indices = torch.tensor([], dtype=torch.long, device=batch.device)
            topk_batch = torch.tensor([], dtype=torch.long, device=batch.device)

        return topk_indices, topk_batch


    def _compute_edge_scores(self, x, edge_index):
        # 使用GAT计算边权重得分
        x_attned, (edge_index_attn, attention_weights) = self.edge_score_conv(x, edge_index, return_attention_weights=True)
        if self.edge_score_heads > 1:
            # if self.gat_concat:
            #     e = self.attention_reduce(attention_weights).squeeze(-1)
            # else:
            #     e = attention_weights.mean(dim=1)
            e = self.attention_reduce(attention_weights).squeeze(-1)
            x_attned = x_attned.view(x.size(0), self.edge_score_heads, self.input_dim).mean(dim=1)
        else:
            e = attention_weights.squeeze(-1)
        # 移除自环
        modified_edge_index, e = remove_self_loops(edge_index_attn, e)
        e = F.dropout(e, p=0.2, training=self.training)
        edge_score = self.compute_edge_score_softmax(e, edge_index, x.size(0))

        return x_attned, modified_edge_index, edge_score

    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Normalizes edge scores via softmax application."""
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)
    
    # def _pool(x, edge_index, batch, global_feature, cluster_center_indices, cluster_center_batch, edge_score):
    #     batch_ids = torch.unique(batch)
        
    #     for batch_id in batch_ids:
    #         mask = (batch == batch_id)
    #         current_x = x[mask]
    #         current_edge_index = edge_index[:, mask]
    #         current_global_feature = global_feature[batch_id]
    #         current_cluster_center_indices = cluster_center_indices[cluster_center_batch == batch_id]
    #         current_edge_score = edge_score[current_edge_index[0], current_edge_index[1]]

    #         # 在这里实现池化操作
    #         # ...
            
    def _pool(self, x, edge_index, batch, global_feature, cluster_center_indices, cluster_center_batch, edge_score):
        batch_ids = torch.unique(batch)
        
        # 存储池化后的结果
        pooled_x_list = []
        pooled_batch_list = []
        # 存储新的边信息
        new_edge_index_list = []
        new_edge_attr_list = []
        
        for batch_id in batch_ids:
            # 获取当前批次的数据
            mask = (batch == batch_id)
            current_x = x[mask]
            current_global_feature = global_feature[batch_id]
            current_cluster_center_indices = cluster_center_indices[cluster_center_batch == batch_id]
            
            # 获取当前批次节点的全局索引
            global_indices = torch.where(mask)[0]
            
            # 创建全局索引到局部索引的映射
            global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(global_indices)}
            
            # 将池化中心节点的全局索引转换为局部索引
            cluster_centers_local = [global_to_local[center_idx.item()] for center_idx in current_cluster_center_indices]
            
            # 初始化簇分配：-1表示未分配，其他值表示分配的簇ID
            cluster_assignment = torch.full((len(current_x),), -1, dtype=torch.long, device=x.device)
            
            # 将池化中心节点分配给自己所在的簇
            for i, center_idx in enumerate(cluster_centers_local):
                cluster_assignment[center_idx] = i
            
            # 获取当前批次的边 - 修正部分
            # 找到所有源节点和目标节点都在当前批次中的边
            edge_mask = torch.isin(edge_index[0], global_indices) & torch.isin(edge_index[1], global_indices)
            current_edge_index = edge_index[:, edge_mask]
            
            # 获取对应的边分数
            current_edge_score = edge_score[edge_mask]
            
            # 将边的全局索引转换为局部索引
            current_edge_index_local = torch.zeros_like(current_edge_index)
            for i in range(current_edge_index.size(1)):
                src_global = current_edge_index[0, i].item()
                dst_global = current_edge_index[1, i].item()
                current_edge_index_local[0, i] = global_to_local[src_global]
                current_edge_index_local[1, i] = global_to_local[dst_global]
            
            # 创建邻接表
            adj_list = [[] for _ in range(len(current_x))]
            for i in range(current_edge_index_local.size(1)):
                src, dst = current_edge_index_local[0, i].item(), current_edge_index_local[1, i].item()
                adj_list[src].append((dst, current_edge_score[i]))
                adj_list[dst].append((src, current_edge_score[i]))
            
            # 为每个池化中心节点合并周围的未分配节点
            for cluster_id, center_idx in enumerate(cluster_centers_local):
                # 使用队列进行BFS，优先合并边分数高的节点
                queue = []
                # 将中心节点的邻居加入队列，按边分数排序
                neighbors = [(neighbor, score) for neighbor, score in adj_list[center_idx] 
                            if cluster_assignment[neighbor] == -1]
                neighbors.sort(key=lambda x: x[1], reverse=True)
                queue.extend(neighbors)
                
                # 处理队列中的节点
                while queue:
                    node, score = queue.pop(0)
                    if cluster_assignment[node] == -1:  # 如果节点未被分配
                        cluster_assignment[node] = cluster_id
                        
                        # 将该节点的未分配邻居加入队列
                        new_neighbors = [(neighbor, min(score, edge_score)) 
                                    for neighbor, edge_score in adj_list[node] 
                                    if cluster_assignment[neighbor] == -1]
                        new_neighbors.sort(key=lambda x: x[1], reverse=True)
                        queue.extend(new_neighbors)
            
            # 对每个簇进行池化（使用edge_score进行加权平均）
            cluster_features = []
            for cluster_id in range(len(cluster_centers_local)):
                cluster_mask = (cluster_assignment == cluster_id)
                if cluster_mask.any():
                    node_ids = torch.arange(cluster_mask.size(0)).to(x.device)
                    cluster_node_ids = node_ids[cluster_mask]
                    cluster_edge_mask = torch.isin(current_edge_index[0], cluster_node_ids) & torch.isin(current_edge_index[1], cluster_node_ids)
                    cluster_edge = current_edge_index[:, cluster_edge_mask]
                    # 获取簇内节点的特征和对应的边分数
                    cluster_features.append(current_x[cluster_mask])
                    # cluster_edge_scores = current_edge_score[cluster_edge_mask]
                    
                    # 使用边分数作为权重进行加权平均
                    # weights = cluster_edge_scores
                    
                    pooled_feature = torch.zeros_like(current_global_feature).to(x.device)
                    for i in range(cluster_edge.size(1)):
                        src = cluster_edge[0, i].item()
                        dst = cluster_edge[1, i].item()
                        # 使用边分数作为权重
                        weight = current_edge_score[cluster_edge_mask][i]
                        pooled_feature += current_x[src] * weight
                        pooled_feature += current_x[dst] * weight

                    # 加权平均
                    # pooled_feature = torch.sum(cluster_features[-1] * weights.unsqueeze(1), dim=0)
                    # 添加全局特征
                    pooled_feature = pooled_feature + current_global_feature
                    
                    pooled_x_list.append(pooled_feature)
                    pooled_batch_list.append(batch_id)
            
            # 处理未被池化的节点
            unassigned_mask = (cluster_assignment == -1)
            if unassigned_mask.any():
                # 为每个未被池化的节点创建一个单独的簇
                unassigned_indices = torch.where(unassigned_mask)[0]
                for idx in unassigned_indices:
                    # 使用节点自身作为池化结果
                    pooled_x_list.append(current_x[idx] + current_global_feature)
                    pooled_batch_list.append(batch_id)
                    # 为未分配节点创建一个新的簇
                    cluster_id = len(cluster_features)
                    cluster_features.append(current_x[idx].unsqueeze(0))
            
            # 构建池化后的边
            num_clusters = len(cluster_features)
            if num_clusters > 1:  # 只有多个簇时才需要构建边
                # 初始化簇间边的权重矩阵
                cluster_adj = torch.zeros((num_clusters, num_clusters), device=x.device)
                cluster_edge_count = torch.zeros((num_clusters, num_clusters), device=x.device)
                
                # 遍历当前批次的所有边
                for i in range(current_edge_index_local.size(1)):
                    src_local = current_edge_index_local[0, i].item()
                    dst_local = current_edge_index_local[1, i].item()
                    
                    src_cluster = cluster_assignment[src_local]
                    dst_cluster = cluster_assignment[dst_local]
                    
                    # 如果边连接的是不同簇的节点
                    if src_cluster != dst_cluster and src_cluster != -1 and dst_cluster != -1:
                        # 累加边权重和计数
                        cluster_adj[src_cluster, dst_cluster] += current_edge_score[i]
                        cluster_adj[dst_cluster, src_cluster] += current_edge_score[i]
                        cluster_edge_count[src_cluster, dst_cluster] += 1
                        cluster_edge_count[dst_cluster, src_cluster] += 1
                
                # 计算平均边权重
                mask = cluster_edge_count > 0
                cluster_adj[mask] = cluster_adj[mask] / cluster_edge_count[mask]
                
                # 构建新的边索引
                rows, cols = torch.where(cluster_adj > 0)
                current_new_edges = torch.stack([rows, cols], dim=0)
                
                # 计算当前批次在池化后节点中的起始索引
                start_idx = len(pooled_x_list) - num_clusters
                
                # 调整边索引以匹配全局索引
                current_new_edges += start_idx
                
                # 获取对应的边权重
                current_edge_weights = cluster_adj[rows, cols]
                
                # 添加到全局边列表中
                new_edge_index_list.append(current_new_edges)
                new_edge_attr_list.append(current_edge_weights)
        
        # 如果没有池化任何节点，返回空结果
        if not pooled_x_list:
            return (torch.tensor([], device=x.device), 
                    torch.tensor([], device=x.device, dtype=torch.long),
                    torch.tensor([], device=x.device, dtype=torch.long),
                    torch.tensor([], device=x.device))
        
        # 合并所有批次的结果
        pooled_x = torch.stack(pooled_x_list)
        pooled_batch = torch.tensor(pooled_batch_list, device=x.device, dtype=torch.long)
        
        # 合并所有边
        if new_edge_index_list:
            new_edge_index = torch.cat(new_edge_index_list, dim=1)
            new_edge_attr = torch.cat(new_edge_attr_list)
        else:
            new_edge_index = torch.tensor([[], []], device=x.device, dtype=torch.long)
            new_edge_attr = torch.tensor([], device=x.device)
        
        return pooled_x, new_edge_index, new_edge_attr, pooled_batch

    def _pool_optimized(self, x, edge_index, batch, global_feature, cluster_center_indices, cluster_center_batch, edge_score):
        batch_ids = torch.unique(batch)
        
        pooled_x_list = []
        pooled_batch_list = []
        new_edge_index_list = []
        new_edge_attr_list = []

        for batch_id in batch_ids:
            mask = (batch == batch_id)
            current_x = x[mask]
            current_global_feature = global_feature[batch_id]
            current_cluster_center_indices = cluster_center_indices[cluster_center_batch == batch_id]

            global_indices = torch.where(mask)[0]
            global_to_local = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
            global_to_local[global_indices] = torch.arange(len(global_indices), device=x.device)

            cluster_centers_local = global_to_local[current_cluster_center_indices]

            cluster_assignment = torch.full((len(current_x),), -1, dtype=torch.long, device=x.device)
            cluster_assignment[cluster_centers_local] = torch.arange(len(cluster_centers_local), device=x.device)

            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            current_edge_index = edge_index[:, edge_mask]
            current_edge_score = edge_score[edge_mask]

            current_edge_index_local = global_to_local[current_edge_index]

            adj_list = [[] for _ in range(len(current_x))]
            for i in range(current_edge_index_local.size(1)):
                src, dst = current_edge_index_local[:, i]
                score = current_edge_score[i]
                adj_list[src.item()].append((dst.item(), score))
                adj_list[dst.item()].append((src.item(), score))

            for cluster_id, center_idx in enumerate(cluster_centers_local):
                visited = torch.zeros(len(current_x), dtype=torch.bool, device=x.device)
                queue = [(n, s) for n, s in adj_list[center_idx.item()] if cluster_assignment[n] == -1]
                queue.sort(key=lambda x: x[1], reverse=True)

                while queue:
                    node, score = queue.pop(0)
                    if cluster_assignment[node] == -1:
                        cluster_assignment[node] = cluster_id
                        visited[node] = True
                        new_neighbors = [(n, min(score, s)) for n, s in adj_list[node] if cluster_assignment[n] == -1 and not visited[n]]
                        new_neighbors.sort(key=lambda x: x[1], reverse=True)
                        queue.extend(new_neighbors)

            cluster_features = []
            for cluster_id in range(len(cluster_centers_local)):
                cluster_mask = (cluster_assignment == cluster_id)
                if cluster_mask.any():
                    pooled_feature = torch.zeros_like(current_global_feature)
                    node_ids = torch.where(cluster_mask)[0]
                    edge_mask = torch.isin(current_edge_index_local[0], node_ids) & torch.isin(current_edge_index_local[1], node_ids)
                    cluster_edge = current_edge_index_local[:, edge_mask]
                    cluster_edge_scores = current_edge_score[edge_mask]

                    for i in range(cluster_edge.size(1)):
                        src, dst = cluster_edge[:, i]
                        weight = cluster_edge_scores[i]
                        pooled_feature += current_x[src] * weight
                        pooled_feature += current_x[dst] * weight

                    pooled_feature += current_global_feature
                    pooled_x_list.append(pooled_feature)
                    pooled_batch_list.append(batch_id)

            unassigned_mask = (cluster_assignment == -1)
            if unassigned_mask.any():
                unassigned_indices = torch.where(unassigned_mask)[0]
                for idx in unassigned_indices:
                    pooled_x_list.append(current_x[idx] + current_global_feature)
                    pooled_batch_list.append(batch_id)
                    cluster_features.append(current_x[idx].unsqueeze(0))

            num_clusters = len(cluster_features)
            if num_clusters > 1:
                cluster_adj = torch.zeros((num_clusters, num_clusters), device=x.device)
                cluster_edge_count = torch.zeros((num_clusters, num_clusters), device=x.device)

                src_cluster = cluster_assignment[current_edge_index_local[0]]
                dst_cluster = cluster_assignment[current_edge_index_local[1]]
                valid_mask = (src_cluster != dst_cluster) & (src_cluster != -1) & (dst_cluster != -1)

                srcs = src_cluster[valid_mask]
                dsts = dst_cluster[valid_mask]
                scores = current_edge_score[valid_mask]

                for i in range(len(srcs)):
                    s, d = srcs[i], dsts[i]
                    cluster_adj[s, d] += scores[i]
                    cluster_adj[d, s] += scores[i]
                    cluster_edge_count[s, d] += 1
                    cluster_edge_count[d, s] += 1

                mask = cluster_edge_count > 0
                cluster_adj[mask] = cluster_adj[mask] / cluster_edge_count[mask]

                rows, cols = torch.where(cluster_adj > 0)
                current_new_edges = torch.stack([rows, cols], dim=0)
                start_idx = len(pooled_x_list) - num_clusters
                current_new_edges += start_idx
                current_edge_weights = cluster_adj[rows, cols]

                new_edge_index_list.append(current_new_edges)
                new_edge_attr_list.append(current_edge_weights)

        if not pooled_x_list:
            return (torch.tensor([], device=x.device), 
                    torch.tensor([], device=x.device, dtype=torch.long),
                    torch.tensor([], device=x.device, dtype=torch.long),
                    torch.tensor([], device=x.device))

        pooled_x = torch.stack(pooled_x_list)
        pooled_batch = torch.tensor(pooled_batch_list, device=x.device, dtype=torch.long)

        if new_edge_index_list:
            new_edge_index = torch.cat(new_edge_index_list, dim=1)
            new_edge_attr = torch.cat(new_edge_attr_list)
        else:
            new_edge_index = torch.tensor([[], []], device=x.device, dtype=torch.long)
            new_edge_attr = torch.tensor([], device=x.device)

        return pooled_x, new_edge_index, new_edge_attr, pooled_batch

    def _pool_optimized_more(self, x, edge_index, batch, global_feature, cluster_center_indices, cluster_center_batch, edge_score):
        from torch_scatter import scatter_add
        from torch_sparse import coalesce
        batch_ids = torch.unique(batch)
        
        pooled_x_list = []
        pooled_batch_list = []
        new_edge_index_list = []
        new_edge_attr_list = []

        for batch_id in batch_ids:
            mask = (batch == batch_id)
            current_x = x[mask]
            current_global_feature = global_feature[batch_id]
            current_cluster_center_indices = cluster_center_indices[cluster_center_batch == batch_id]

            global_indices = torch.where(mask)[0]
            global_to_local = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
            global_to_local[global_indices] = torch.arange(len(global_indices), device=x.device)

            cluster_centers_local = global_to_local[current_cluster_center_indices]

            cluster_assignment = torch.full((len(current_x),), -1, dtype=torch.long, device=x.device)
            cluster_assignment[cluster_centers_local] = torch.arange(len(cluster_centers_local), device=x.device)

            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            current_edge_index = edge_index[:, edge_mask]
            current_edge_score = edge_score[edge_mask]
            current_edge_index_local = global_to_local[current_edge_index]

            # 构建稀疏邻接矩阵
            adj_edge_index, adj_edge_weight = coalesce(current_edge_index_local, current_edge_score, m=len(current_x), n=len(current_x))

            # 向量化传播簇标签（模拟 BFS）
            for _ in range(3):  # 可调传播步数
                src, dst = adj_edge_index
                src_cluster = cluster_assignment[src]
                dst_cluster = cluster_assignment[dst]
                mask = (src_cluster != -1) & (dst_cluster == -1)
                cluster_assignment[dst[mask]] = src_cluster[mask]

            num_clusters = cluster_assignment.max().item() + 1 if cluster_assignment.max().item() >= 0 else 0

            # 边权加权聚合特征
            src, dst = adj_edge_index
            src_cluster = cluster_assignment[src]
            dst_cluster = cluster_assignment[dst]
            valid_mask = (src_cluster != -1) & (dst_cluster != -1)

            src_feat = current_x[src[valid_mask]] * adj_edge_weight[valid_mask].unsqueeze(1)
            dst_feat = current_x[dst[valid_mask]] * adj_edge_weight[valid_mask].unsqueeze(1)
            all_feat = torch.cat([src_feat, dst_feat], dim=0)
            all_cluster = torch.cat([src_cluster[valid_mask], dst_cluster[valid_mask]], dim=0)

            pooled_x = scatter_add(all_feat, all_cluster, dim=0, dim_size=num_clusters)
            pooled_x += current_global_feature  # 广播加全局特征

            pooled_x_list.append(pooled_x)
            pooled_batch_list.extend([batch_id] * num_clusters)

            # 未分配节点处理
            unassigned_mask = (cluster_assignment == -1)
            if unassigned_mask.any():
                unassigned_x = current_x[unassigned_mask] + current_global_feature
                pooled_x_list.append(unassigned_x)
                pooled_batch_list.extend([batch_id] * unassigned_x.size(0))
                cluster_assignment[unassigned_mask] = torch.arange(num_clusters, num_clusters + unassigned_x.size(0), device=x.device)
                num_clusters += unassigned_x.size(0)

            # 构建簇间边
            src_cluster = cluster_assignment[src]
            dst_cluster = cluster_assignment[dst]
            inter_mask = (src_cluster != dst_cluster) & (src_cluster != -1) & (dst_cluster != -1)

            inter_src = src_cluster[inter_mask]
            inter_dst = dst_cluster[inter_mask]
            inter_score = adj_edge_weight[inter_mask]

            edge_weight_matrix = torch.zeros((num_clusters, num_clusters), device=x.device)
            edge_count_matrix = torch.zeros_like(edge_weight_matrix)

            edge_weight_matrix.index_put_((inter_src, inter_dst), inter_score, accumulate=True)
            edge_weight_matrix.index_put_((inter_dst, inter_src), inter_score, accumulate=True)
            edge_count_matrix.index_put_((inter_src, inter_dst), torch.ones_like(inter_score), accumulate=True)
            edge_count_matrix.index_put_((inter_dst, inter_src), torch.ones_like(inter_score), accumulate=True)

            mask = edge_count_matrix > 0
            edge_weight_matrix[mask] /= edge_count_matrix[mask]

            rows, cols = torch.where(edge_weight_matrix > 0)
            current_new_edges = torch.stack([rows, cols], dim=0)
            start_idx = len(pooled_batch_list) - num_clusters
            current_new_edges += start_idx
            current_edge_weights = edge_weight_matrix[rows, cols]

            new_edge_index_list.append(current_new_edges)
            new_edge_attr_list.append(current_edge_weights)

        if not pooled_x_list:
            return (torch.tensor([], device=x.device), 
                    torch.tensor([], device=x.device, dtype=torch.long),
                    torch.tensor([], device=x.device, dtype=torch.long),
                    torch.tensor([], device=x.device))

        pooled_x = torch.cat(pooled_x_list, dim=0)
        pooled_batch = torch.tensor(pooled_batch_list, device=x.device, dtype=torch.long)

        if new_edge_index_list:
            new_edge_index = torch.cat(new_edge_index_list, dim=1)
            new_edge_attr = torch.cat(new_edge_attr_list)
        else:
            new_edge_index = torch.tensor([[], []], device=x.device, dtype=torch.long)
            new_edge_attr = torch.tensor([], device=x.device)

        return pooled_x, new_edge_index, new_edge_attr, pooled_batch

    def _pool_parallel_more(self, x, edge_index, batch, global_feature, cluster_center_indices, cluster_center_batch, edge_score):
        from torch_scatter import scatter_add
        from torch_sparse import coalesce
        import torch
        num_nodes = x.size(0)
        device = x.device

        # Step 1: 构建全局映射
        global_to_local = torch.arange(num_nodes, device=device)

        # Step 2: 初始化簇分配
        cluster_assignment = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        cluster_ids = torch.arange(cluster_center_indices.size(0), device=device)
        cluster_assignment[cluster_center_indices] = cluster_ids

        # Step 3: 构建稀疏邻接矩阵
        edge_index, edge_score = coalesce(edge_index, edge_score, m=num_nodes, n=num_nodes)

        # Step 4: 向量化传播簇标签（模拟 BFS）
        for _ in range(3):  # 可调传播步数
            src, dst = edge_index
            src_cluster = cluster_assignment[src]
            dst_cluster = cluster_assignment[dst]
            mask = (src_cluster != -1) & (dst_cluster == -1)
            cluster_assignment[dst[mask]] = src_cluster[mask]

        # Step 5: 处理未分配节点（每个节点独立成簇）
        unassigned_mask = (cluster_assignment == -1)
        unassigned_indices = torch.where(unassigned_mask)[0]
        new_cluster_ids = torch.arange(cluster_ids.size(0), cluster_ids.size(0) + unassigned_indices.size(0), device=device)
        cluster_assignment[unassigned_indices] = new_cluster_ids

        # Step 6: 聚合特征（边权加权 + 全局特征）
        src, dst = edge_index
        src_cluster = cluster_assignment[src]
        dst_cluster = cluster_assignment[dst]
        valid_mask = (src_cluster != -1) & (dst_cluster != -1)

        src_feat = x[src[valid_mask]] * edge_score[valid_mask].unsqueeze(1)
        dst_feat = x[dst[valid_mask]] * edge_score[valid_mask].unsqueeze(1)
        all_feat = torch.cat([src_feat, dst_feat], dim=0)
        all_cluster = torch.cat([src_cluster[valid_mask], dst_cluster[valid_mask]], dim=0)

        num_clusters = cluster_assignment.max().item() + 1
        pooled_x = scatter_add(all_feat, all_cluster, dim=0, dim_size=num_clusters)

        # Step 7: 添加全局特征
        cluster_batch = batch[cluster_center_indices]
        full_cluster_batch = torch.full((num_clusters,), -1, dtype=torch.long, device=device)
        full_cluster_batch[cluster_assignment[cluster_center_indices]] = cluster_batch
        full_cluster_batch[cluster_assignment[unassigned_indices]] = batch[unassigned_indices]
        pooled_x += global_feature[full_cluster_batch]

        # Step 8: 构建簇间边
        inter_mask = (src_cluster != dst_cluster)
        inter_src = src_cluster[inter_mask]
        inter_dst = dst_cluster[inter_mask]
        inter_score = edge_score[inter_mask]

        edge_pairs = inter_src * num_clusters + inter_dst
        edge_weights = scatter_add(inter_score, edge_pairs, dim=0, dim_size=num_clusters * num_clusters)
        edge_counts = scatter_add(torch.ones_like(inter_score), edge_pairs, dim=0, dim_size=num_clusters * num_clusters)

        edge_weights = edge_weights.view(num_clusters, num_clusters)
        edge_counts = edge_counts.view(num_clusters, num_clusters)
        edge_weights[edge_counts > 0] /= edge_counts[edge_counts > 0]

        rows, cols = torch.where(edge_weights > 0)
        new_edge_index = torch.stack([rows, cols], dim=0)
        new_edge_attr = edge_weights[rows, cols]

        # Step 9: 构建 batch 信息
        pooled_batch = full_cluster_batch

        return pooled_x, new_edge_index, new_edge_attr, pooled_batch

    def _pool_parallel(self, x, edge_index, batch, global_feature, cluster_center_indices, cluster_center_batch, edge_score):
        num_nodes = x.size(0)
        device = x.device

        # Step 1: 初始化簇分配
        cluster_assignment = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        cluster_ids = torch.arange(cluster_center_indices.size(0), device=device)
        cluster_assignment[cluster_center_indices] = cluster_ids

        # Step 2: 构建稀疏邻接矩阵
        edge_index, edge_score = coalesce(edge_index, edge_score, m=num_nodes, n=num_nodes)

        # Step 3: 向量化传播簇标签（模拟 BFS）
        for _ in range(3):
            src, dst = edge_index
            src_cluster = cluster_assignment[src]
            dst_cluster = cluster_assignment[dst]
            mask = (src_cluster != -1) & (dst_cluster == -1)
            cluster_assignment[dst[mask]] = src_cluster[mask]

        # Step 4: 处理未分配节点
        unassigned_mask = (cluster_assignment == -1)
        unassigned_indices = torch.where(unassigned_mask)[0]
        new_cluster_ids = torch.arange(cluster_ids.size(0), cluster_ids.size(0) + unassigned_indices.size(0), device=device)
        cluster_assignment[unassigned_indices] = new_cluster_ids

        # Step 5: 聚合特征（使用 segment_csr）
        num_clusters = cluster_assignment.max().item() + 1
        src, dst = edge_index
        src_cluster = cluster_assignment[src]
        dst_cluster = cluster_assignment[dst]
        valid_mask = (src_cluster != -1) & (dst_cluster != -1)

        all_nodes = torch.cat([src[valid_mask], dst[valid_mask]])
        all_clusters = torch.cat([src_cluster[valid_mask], dst_cluster[valid_mask]])
        all_weights = torch.cat([edge_score[valid_mask], edge_score[valid_mask]])

        # 构建 CSR 索引
        sorted_clusters, sorted_idx = all_clusters.sort()
        sorted_nodes = all_nodes[sorted_idx]
        sorted_weights = all_weights[sorted_idx]

        cluster_counts = torch.bincount(sorted_clusters, minlength=num_clusters)
        cluster_ptr = torch.zeros(num_clusters + 1, dtype=torch.int32, device=device)
        cluster_ptr[1:] = torch.cumsum(cluster_counts, dim=0)

        pooled_x = torch.ops.pytorch.segment_csr(
            x[sorted_nodes] * sorted_weights.unsqueeze(1),
            cluster_ptr,
            reduce="sum"
        )

        # Step 6: 添加全局特征（使用缓存索引）
        center_cluster_ids = cluster_assignment[cluster_center_indices]
        unassigned_cluster_ids = cluster_assignment[unassigned_indices]

        full_cluster_batch = torch.full((num_clusters,), -1, dtype=torch.long, device=device)
        full_cluster_batch[center_cluster_ids] = cluster_center_batch
        full_cluster_batch[unassigned_cluster_ids] = batch[unassigned_indices]

        pooled_x += global_feature[full_cluster_batch]

        # Step 7: 构建簇间边
        inter_mask = (src_cluster != dst_cluster)
        inter_src = src_cluster[inter_mask]
        inter_dst = dst_cluster[inter_mask]
        inter_score = edge_score[inter_mask]

        edge_pairs = inter_src * num_clusters + inter_dst
        edge_weights = torch.zeros(num_clusters * num_clusters, device=device)
        edge_counts = torch.zeros_like(edge_weights)

        edge_weights.index_add_(0, edge_pairs, inter_score)
        edge_counts.index_add_(0, edge_pairs, torch.ones_like(inter_score))

        edge_weights[edge_counts > 0] /= edge_counts[edge_counts > 0]
        edge_weights = edge_weights.view(num_clusters, num_clusters)

        rows, cols = torch.where(edge_weights > 0)
        new_edge_index = torch.stack([rows, cols], dim=0)
        new_edge_attr = edge_weights[rows, cols]

        pooled_batch = full_cluster_batch

        return pooled_x, new_edge_index, new_edge_attr, pooled_batch
