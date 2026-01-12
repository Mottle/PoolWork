# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_sparse import spspmm
# from torch_geometric.nn import global_mean_pool


# class H2GCN(nn.Module):
#     def __init__(self, feat_dim, hidden_dim, k=2, dropout=0.5, use_relu=True):
#         super().__init__()
#         self.push_pe = None
#         self.k = k
#         self.dropout = dropout
#         self.act = F.relu if use_relu else lambda x: x

#         self.w_embed = nn.Parameter(torch.zeros(feat_dim, hidden_dim))
#         # 不再在这里做分类，作为纯 encoder / pooler
#         # self.reset_parameters()

#         # self.initialized = False
#         # self.A1 = None
#         # self.A2 = None

#         h2_out_dim = (2 ** (k + 1) - 1) * hidden_dim
#         self.out_linear = nn.Linear(h2_out_dim, hidden_dim)

#     def reset_parameters(self):
#         pass
#         # nn.init.xavier_uniform_(self.w_embed)

#     def normalize(self, row, col, val, num_nodes):
#         deg = torch.zeros(num_nodes, device=row.device).scatter_add_(0, row, val)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
#         val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
#         return row, col, val

#     def prepare_prop(self, edge_index, num_nodes):
#         device = edge_index.device
#         row, col = edge_index

#         # A1: 1-hop, no self-loop
#         mask = row != col
#         rowA1 = row[mask]
#         colA1 = col[mask]
#         valA1 = torch.ones_like(rowA1, dtype=torch.float)

#         # A2: 2-hop neighbors, remove 1-hop and self-loop
#         indexA = torch.stack([row, col], dim=0)
#         valA = torch.ones(row.size(0), device=device)

#         indexA2, valA2 = spspmm(
#             indexA, valA,
#             indexA, valA,
#             num_nodes, num_nodes, num_nodes
#         )
#         rowA2, colA2 = indexA2

#         # remove self-loops
#         mask2 = rowA2 != colA2
#         rowA2 = rowA2[mask2]
#         colA2 = colA2[mask2]

#         # remove 1-hop edges
#         A1_pairs = set(zip(rowA1.tolist(), colA1.tolist()))
#         A2_pairs = set(zip(rowA2.tolist(), colA2.tolist()))
#         A2_only = A2_pairs - A1_pairs

#         if len(A2_only) > 0:
#             rowA2, colA2 = zip(*A2_only)
#             rowA2 = torch.tensor(rowA2, device=device, dtype=torch.long)
#             colA2 = torch.tensor(colA2, device=device, dtype=torch.long)
#         else:
#             rowA2 = torch.tensor([], device=device, dtype=torch.long)
#             colA2 = torch.tensor([], device=device, dtype=torch.long)

#         valA2 = torch.ones_like(rowA2, dtype=torch.float)

#         self.A1 = self.normalize(rowA1, colA1, valA1, num_nodes)
#         self.A2 = self.normalize(rowA2, colA2, valA2, num_nodes)
#         # self.initialized = True

#     def forward(self, x, edge_index, batch, *args):
#         num_nodes = x.size(0)
#         # if not self.initialized:
#         #     self.prepare_prop(edge_index, num_nodes)

#         self.prepare_prop(edge_index, num_nodes)

#         row1, col1, val1 = self.A1
#         row2, col2, val2 = self.A2

#         # node-level embedding
#         r0 = self.act(x @ self.w_embed)
#         rs = [r0]

#         for _ in range(self.k):
#             r_last = rs[-1]

#             r1 = torch.zeros_like(r_last)
#             r1.index_add_(0, row1, val1.unsqueeze(1) * r_last[col1])

#             r2 = torch.zeros_like(r_last)
#             r2.index_add_(0, row2, val2.unsqueeze(1) * r_last[col2])

#             rs.append(self.act(torch.cat([r1, r2], dim=1)))

#         h_node = torch.cat(rs, dim=1)              # [num_nodes, hidden_all]
#         h_node = F.dropout(h_node, self.dropout, training=self.training)

#         # graph-level pooling
#         feat = global_mean_pool(h_node, batch)   # [num_graphs, hidden_all]
#         feat = self.out_linear(feat)

#         additional_loss = torch.tensor(0.0, device=x.device)
#         return feat, additional_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm, coalesce
from torch_geometric.nn import global_mean_pool


class H2GCN(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, k: int = 2,
                 dropout: float = 0.5, use_relu: bool = True,
                 project_out: bool = True):
        super().__init__()
        self.k = k
        self.dropout = dropout
        self.act = F.relu if use_relu else lambda x: x
        self.push_pe = None

        # 节点初始线性映射 X -> r0
        self.w_embed = nn.Parameter(torch.zeros(feat_dim, hidden_dim))

        # H2GCN 节点表示维度：(2^{k+1} - 1) * hidden_dim
        self.hidden_dim = hidden_dim
        self.out_dim = (2 ** (k + 1) - 1) * hidden_dim

        # 可选：把高维 concat 压回 hidden_dim，方便后面 classifier 对接
        self.project_out = project_out
        if project_out:
            self.out_linear = nn.Linear(self.out_dim, hidden_dim)
        else:
            self.out_linear = None

        self.reset_parameters()

    def reset_parameters(self):
        pass
        # nn.init.xavier_uniform_(self.w_embed)
        # if self.out_linear is not None:
        #     nn.init.xavier_uniform_(self.out_linear.weight)
        #     if self.out_linear.bias is not None:
        #         nn.init.zeros_(self.out_linear.bias)

    # -----------------------------
    # 归一化 D^{-1/2} A D^{-1/2}
    # row, col, val 表示 COO
    # -----------------------------
    @staticmethod
    def normalize(row, col, val, num_nodes):
        device = row.device
        deg = torch.zeros(num_nodes, device=device).scatter_add_(0, row, val)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
        return row, col, val

    # -----------------------------
    # 构造 A1 = indicator(A - I)
    # 和 A2 = indicator(A^2 - A - I)
    # 完全稀疏 + GPU + 无 Python set
    # -----------------------------
    def prepare_prop(self, edge_index, num_nodes):
        device = edge_index.device
        row, col = edge_index  # [E]

        # A: 用 COO (indexA, valA) 表示
        indexA = torch.stack([row, col], dim=0)                # [2, E]
        valA = torch.ones(row.size(0), device=device)          # [E]

        # I: 单位矩阵
        diag = torch.arange(num_nodes, device=device, dtype=torch.long)
        indexI = torch.stack([diag, diag], dim=0)              # [2, N]
        valI = torch.ones(num_nodes, device=device)            # [N]

        # ---------- A1 = indicator(A - I) ----------
        # 构造 (A - I) 的 COO：index = [A, I], val = [1, -1]
        index_A1_raw = torch.cat([indexA, indexI], dim=1)      # [2, E+N]
        val_A1_raw = torch.cat([valA, -valI], dim=0)           # [E+N]

        index_A1, val_A1 = coalesce(
            index_A1_raw, val_A1_raw,
            num_nodes, num_nodes
        )
        # 只保留 >0 的项 ⇒ (A - I) > 0 对应的边
        mask_A1 = val_A1 > 0
        rowA1 = index_A1[0, mask_A1]
        colA1 = index_A1[1, mask_A1]
        valA1 = torch.ones_like(rowA1, dtype=torch.float, device=device)

        # ---------- A2 = indicator(A^2 - A - I) ----------
        # 先算 A^2
        indexA2, valA2 = spspmm(
            indexA, valA,
            indexA, valA,
            num_nodes, num_nodes, num_nodes
        )   # indexA2: [2, E2], valA2: [E2]

        # 构造 (A^2 - A - I) 的 COO: index = [A2, A, I], val = [1, -1, -1]
        index_A2_raw = torch.cat([indexA2, indexA, indexI], dim=1)
        val_A2_raw = torch.cat([
            torch.ones(indexA2.size(1), device=device),
            -valA,
            -valI
        ], dim=0)

        index_A2, val_A2 = coalesce(
            index_A2_raw, val_A2_raw,
            num_nodes, num_nodes
        )
        # >0 的条目就是 A^2 - A - I 中仍然存在的边
        mask_A2 = val_A2 > 0
        rowA2 = index_A2[0, mask_A2]
        colA2 = index_A2[1, mask_A2]
        valA2 = torch.ones_like(rowA2, dtype=torch.float, device=device)

        # 归一化
        self.A1 = self.normalize(rowA1, colA1, valA1, num_nodes)
        self.A2 = self.normalize(rowA2, colA2, valA2, num_nodes)

    # -----------------------------
    # 稀疏乘法：Y = Â X
    # row, col, val 表示归一化后的 COO
    # X: [N, d] -> Y: [N, d]
    # -----------------------------
    @staticmethod
    def spmm(row, col, val, x, num_nodes):
        out = torch.zeros_like(x)
        out.index_add_(0, row, val.unsqueeze(-1) * x[col])
        return out

    # -----------------------------
    # forward: 图分类版 H2GCN
    # 返回：graph-level embedding, additional_loss
    # -----------------------------
    def forward(self, x, edge_index, batch, *args):
        """
        x: [num_nodes, feat_dim]
        edge_index: [2, num_edges]
        batch: [num_nodes]，图索引
        """
        num_nodes = x.size(0)
        device = x.device

        # 每个 batch 单独构造 A1, A2
        self.prepare_prop(edge_index, num_nodes)

        row1, col1, val1 = self.A1
        row2, col2, val2 = self.A2

        # r0
        r0 = self.act(x @ self.w_embed)     # [N, hidden_dim]
        rs = [r0]

        # k 层 H2GCN 传播
        for _ in range(self.k):
            r_last = rs[-1]

            r1 = self.spmm(row1, col1, val1, r_last, num_nodes)
            r2 = self.spmm(row2, col2, val2, r_last, num_nodes)

            rs.append(self.act(torch.cat([r1, r2], dim=1)))

        # concat 所有层的节点表示
        h_node = torch.cat(rs, dim=1)       # [N, out_dim]
        h_node = F.dropout(h_node, self.dropout, training=self.training)

        # 图级池化
        graph_feat = global_mean_pool(h_node, batch)  # [num_graphs, out_dim]

        if self.project_out:
            graph_feat = self.out_linear(graph_feat)  # [num_graphs, hidden_dim]

        additional_loss = torch.tensor(0.0, device=device)
        return graph_feat, additional_loss
