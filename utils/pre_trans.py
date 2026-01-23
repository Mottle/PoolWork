import torch
from torch_geometric.utils import to_dense_adj

import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import AddRandomWalkPE

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

def pre_transform_all(
    spd_threshold=80,
    spd_k_max=4,
    lpe_dim=20,
    verbose=False
):
    spd_fn = compute_spd_hybrid_edge(threshold=spd_threshold, k_max=spd_k_max, verbose=verbose)
    deg_fn = compute_degree
    wpe = AddRandomWalkPE(walk_length=20, attr_name='rw_pos_enc')

    def transform(data):
        if verbose:
            print(f"\n=== Pre-transform graph with {data.num_nodes} nodes ===")

        data = deg_fn(data, verbose=verbose)
        data = spd_fn(data)
        data = wpe(data)

        if verbose:
            print(f"=== Pre-transform done ===\n")

        return data

    return transform
