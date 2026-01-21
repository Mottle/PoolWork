import torch

def compute_Ap_Up(edge_index, num_nodes, P):
    """
    计算 A^p 和 U_p = A^p - A^(p-1)
    返回:
        A_p_list: 长度 P 每个元素是 {'idx': Tensor[2, nnz], 'wei': Tensor[nnz]}
        U_p_list: 同上
    """
    device = edge_index.device

    # A (sparse adjacency)
    A_sparse = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.size(1), device=device),
        (num_nodes, num_nodes)
    ).coalesce()

    # A^0 = I
    idx = torch.arange(num_nodes, device=device)
    A_prev = torch.sparse_coo_tensor(
        torch.stack([idx, idx], dim=0),
        torch.ones(num_nodes, device=device),
        (num_nodes, num_nodes)
    ).coalesce()

    Ap = A_sparse

    A_p_list = []
    U_p_list = []

    for _ in range(1, P + 1):
        # ---- 保存 A^p ----
        Ap_cpu = {
            'idx': Ap.indices().cpu(),
            'wei': Ap.values().cpu()
        }
        A_p_list.append(Ap_cpu)

        # ---- 计算 U_p = A^p - A^(p-1) ----
        U_p = (Ap - A_prev).coalesce()
        U_p_cpu = {
            'idx': U_p.indices().cpu(),
            'wei': U_p.values().cpu()
        }
        U_p_list.append(U_p_cpu)

        # ---- 更新 A^(p-1) 和 A^p ----
        A_prev = Ap
        Ap = torch.sparse.mm(Ap, A_sparse).coalesce()

    return A_p_list, U_p_list

def compute_Ap_Up_optimized(edge_index, num_nodes, P):
    device = torch.device('cpu') 
    edge_index = edge_index.to(device)
    
    # 2. 构造初始矩阵 (Value = 1.0, 纯路径计数)
    A_base = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.size(1), dtype=torch.float32, device=device),
        (num_nodes, num_nodes)
    ).coalesce()
    
    Ap = A_base      # 当前 A^p
    A_prev = None    # 上一跳 A^(p-1)
    
    A_p_list = []
    U_p_list = []

    for p in range(1, P + 1):
        if p == 1:
            Up = Ap
        else:
            Up = (Ap - A_prev).coalesce()
            mask = Up.values() != 0
            if not mask.all():
                Up = torch.sparse_coo_tensor(
                    Up.indices()[:, mask],
                    Up.values()[mask],
                    Up.size()
                ).coalesce()

        # 1. 存储结果 (Clone 防止引用覆盖)
        A_p_list.append({
            'idx': Ap.indices().clone(),
            'wei': Ap.values().clone()
        })
        
        U_p_list.append({
            'idx': Up.indices().clone(),
            'wei': Up.values().clone()
        })

        # 2. 迭代更新 (准备下一次循环)
        if p < P:
            A_prev = Ap
            
            # 计算下一跳 A^(p+1) = A^p @ A_base
            # 注意：未归一化时，这里的值会变成 1, 5, 25, 100... 增长极快
            Ap = torch.sparse.mm(Ap, A_base).coalesce()
            
    return A_p_list, U_p_list