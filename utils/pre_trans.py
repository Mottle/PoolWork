from turtle import forward
import torch
from torch_geometric.utils import to_dense_adj

import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import AddRandomWalkPE
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, degree, dense_to_sparse
from torch_scatter import scatter_add

def compute_spd_hybrid_edge(threshold=80, k_max=4, verbose=False):
    """
    SPD 预处理（混合 Floyd + 稀疏矩阵乘法）
    输出 edge-level SPD（避免 PyG collate 报错）
    """
    def log(msg):
        if verbose:
            print(f"[SPD] {msg}")

    def transform(data):
        N = data.num_nodes
        log(f"Processing graph with {N} nodes")

        # Dense adjacency
        A = to_dense_adj(data.edge_index, max_num_nodes=N)[0]
        A = (A > 0).float()

        # 初始化 SPD dense（临时）
        spd = torch.full((N, N), fill_value=k_max, dtype=torch.long)
        spd[torch.arange(N), torch.arange(N)] = 0

        # -----------------------------
        # 方案 A：小图 → Floyd–Warshall
        # -----------------------------
        if N <= threshold:
            log(f"Using Floyd–Warshall (N={N} ≤ {threshold})")

            dist = torch.full((N, N), 999, dtype=torch.long)
            dist[A > 0] = 1
            dist[torch.arange(N), torch.arange(N)] = 0

            for k in range(N):
                dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k].unsqueeze(0))

            spd = torch.clamp(dist, max=k_max)
            log(f"Floyd done. Max SPD = {spd.max().item()}")

        # ---------------------------------------
        # 方案 B：大图 → 稀疏矩阵乘法（Graphormer）
        # ---------------------------------------
        else:
            log(f"Using sparse-matrix multiplication (N={N} > {threshold})")

            reach = A.clone()

            for k in range(1, k_max + 1):
                newly_reached = (reach > 0) & (spd == k_max)
                spd[newly_reached] = k
                reach = (reach @ A).clamp(max=1)

            log(f"Sparse-matmul done. Max SPD = {spd.max().item()}")

        # -----------------------------
        # 关键：只保存 edge-level SPD
        # -----------------------------
        row, col = data.edge_index
        spd_edge = spd[row, col]  # [E]

        data.spd_index = data.edge_index
        data.spd_attr = spd_edge.long()

        # 不保存 dense SPD（避免 PyG collate 报错）
        # if hasattr(data, "spd"):
        #     del data.spd

        log(f"Saved edge-level SPD for {spd_edge.numel()} edges")

        return data

    return transform


def compute_degree(data, verbose=False):
    if verbose:
        print(f"[DEG] Computing degree for graph with {data.num_nodes} nodes")

    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
    data.degree = deg.long()

    if verbose:
        print(f"[DEG] Done. Max degree = {deg.max().item()}")
    return data

class AddGRITPE(T.BaseTransform):
    """
    计算 GRIT 模型所需的 Random Walk Positional Encoding (RWPE)。
    
    输出:
        data.rrwp_abs: [Num_Nodes, walk_length]  (节点级: RWSE)
        data.rrwp:     [Num_Edges, walk_length]  (边级: Relative RW probability)
    """
    def __init__(self, walk_length=20, attr_name_abs='rrwp_abs', attr_name_rel='rrwp'):
        self.walk_length = walk_length
        self.attr_name_abs = attr_name_abs
        self.attr_name_rel = attr_name_rel

    def forward(self, data):
        return self.__call__(data)

    def __call__(self, data):
        # 1. 获取基本信息
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        device = edge_index.device

        # 2. 构造稠密邻接矩阵 (A)
        # NCI1 等分子图节点少 (<100)，用 Dense 计算最快且方便
        # [N, N]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

        # 3. 计算随机游走转移矩阵 (P = D^-1 * A)
        # 计算度数
        deg = adj.sum(dim=1) # [N]
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0 # 处理孤立点
        
        # P = D^-1 @ A
        # 利用广播机制: column_j * row_i_inv
        P = adj * deg_inv.view(-1, 1) 

        # 4. 迭代计算 P^k 并收集数据
        # 我们需要收集 k=1 到 k=walk_length 的概率
        
        pe_list_abs = [] # 用于存储对角线 (N, K)
        pe_list_rel = [] # 用于存储边 (E, K)
        
        # 用于提取边上概率的索引
        row, col = edge_index
        
        # P_k 初始化为 P^1
        P_k = P.clone()

        for k in range(self.walk_length):
            # --- (A) 收集 Absolute PE (对角线) ---
            # 取对角线元素: P_k[i, i]
            diag = P_k.diagonal()
            pe_list_abs.append(diag)

            # --- (B) 收集 Relative PE (边) ---
            # 只取 data.edge_index 存在的那些边的概率值
            # P_k[row, col]
            rel_val = P_k[row, col]
            pe_list_rel.append(rel_val)

            # --- (C) 迭代: P^{k+1} = P^k @ P ---
            # 只有最后一步不需要乘
            if k < self.walk_length - 1:
                P_k = torch.mm(P_k, P)

        # 5. 堆叠并保存到 data
        # [N, K]
        tensor_abs = torch.stack(pe_list_abs, dim=-1)
        # [E, K]
        tensor_rel = torch.stack(pe_list_rel, dim=-1)

        # 赋值
        setattr(data, self.attr_name_abs, tensor_abs)
        setattr(data, self.attr_name_rel, tensor_rel)

        return data

def pre_transform_all(
    spd_threshold=80,
    spd_k_max=4,
    lpe_dim=20,
    verbose=False
):
    spd_fn = compute_spd_hybrid_edge(threshold=spd_threshold, k_max=spd_k_max, verbose=verbose)
    deg_fn = compute_degree
    # wpe = AddRandomWalkPE(walk_length=20, attr_name='rw_pos_enc')
    wpe_and_rrwp = AddGRITPE()

    def transform(data):
        if verbose:
            print(f"\n=== Pre-transform graph with {data.num_nodes} nodes ===")

        data = deg_fn(data, verbose=verbose)
        data = spd_fn(data)
        data = wpe_and_rrwp(data)

        if verbose:
            print(f"=== Pre-transform done ===\n")

        return data

    return transform
