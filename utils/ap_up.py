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



def compute_Ap_Up_REDDIT_B(edge_index, num_nodes, P):
    # REDDIT-B 的图通常在 1000-4000 节点之间
    # 对于这种规模，如果图变得稠密，使用 Dense Tensor 计算比 Sparse COO 更省内存
    # 阈值可调，通常 10000 节点以下用 Dense 都在 CPU 承受范围内
    
    device = torch.device('cpu') 
    
        # --- Dense 模式 (省内存关键) ---
    from torch_geometric.utils import to_dense_adj
    
    # 构造稠密邻接矩阵 [N, N]
    # 注意：这里我们只用 edge_index 构造 0/1 矩阵，如果是加权图需传入 edge_attr
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    
    Ap = adj.clone()
    # 用 bool 矩阵记录已访问的路径，避免数值无限增大溢出 float (如果只需要结构)
    # 如果你需要精确的路径数量计数，请保持 float，但要注意数值爆炸
    visited_mask = (adj > 0) 
    
    A_p_list = []
    U_p_list = []
    
    # 存入第一跳
    # 为了节省存储空间，最终结果我们还是存为 sparse_coo
    A_p_list.append(dense_to_sparse_dict(Ap))
    U_p_list.append(dense_to_sparse_dict(Ap)) # U1 = A1
    
    cur_A = Ap
    prev_A_mask = visited_mask.clone()
    
    for p in range(2, P + 1):
        # 核心矩阵乘法：Dense MM 极快且内存占用固定
        # A^p = A^(p-1) @ A^1
        cur_A = torch.mm(cur_A, adj)
        
        # --- 计算 Up (Unique Paths at hop p) ---
        # 定义：在第 p 跳新到达的，且在 1...p-1 跳未曾到达过的
        # 这里逻辑取决于你 Up 的定义：
        # 定义A: Up = A^p - A^(p-1) (数值相减，保留路径计数差值)
        # 定义B: Up = (A^p > 0) & ~(Visited) (仅保留新到达的连接结构)
        
        # 你的原代码逻辑倾向于数值相减。但在社交网络中，路径数增长是指数级的(1, 10, 100...)
        # 建议：如果只是为了 GNN 结构特征，只保留结构 (mask) 会更稳定。
        # 下面保留你的原逻辑：数值相减，但利用 mask 过滤零值
        
        # 1. 找出当前 A^p 中非零的位置
        curr_mask = (cur_A > 0)
        
        # 2. Up = A^p 中 那些 (之前没访问过) 的位置
        # 如果你要保留数值：
        Up_dense = torch.zeros_like(cur_A)
        # 仅在“之前没去过”且“现在能去”的地方保留数值
        new_visit_mask = curr_mask & (~prev_A_mask)
        Up_dense[new_visit_mask] = cur_A[new_visit_mask]
        
        # 更新已访问掩码
        prev_A_mask = prev_A_mask | curr_mask
        
        # 3. 结果转稀疏存储 (减少内存占用)
        A_p_list.append(dense_to_sparse_dict(cur_A))
        U_p_list.append(dense_to_sparse_dict(Up_dense))

    return A_p_list, U_p_list

def dense_to_sparse_dict(dense_tensor):
    indices = torch.nonzero(dense_tensor, as_tuple=False).t()
    values = dense_tensor[indices[0], indices[1]]
    return {
        'idx': indices,
        'wei': values
    }