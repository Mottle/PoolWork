import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class StrictBatchSheafConv(MessagePassing):
    def __init__(self, channels, stalk_dim, setup_dropout=0.2):
        """
        channels: 输入/输出总特征维度 (C)
        stalk_dim: Stalk 的维度 (d)
        """
        super(StrictBatchSheafConv, self).__init__(aggr='add')
        assert channels % stalk_dim == 0
        self.d = stalk_dim
        self.k = channels // stalk_dim
        self.channels = channels

        # 1. Sheaf Learner: 生成边上的限制映射
        # 严格定义：每条边需要生成 F_ue 和 F_ve (均为 d*d)
        # 我们使用 linear + matrix_exp 确保映射在正交群 O(d) 中
        self.map_generator = nn.Sequential(
            nn.Linear(2 * channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.d * self.d) 
        )
        
        # 2. 可学习的扩散步长
        self.eps = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(setup_dropout)

    def forward(self, x, edge_index):
        # x: [Total_Nodes, C]
        # edge_index: [2, Total_Edges] (PyG Batch 会自动处理索引偏移)
        
        row, col = edge_index
        
        # 生成映射矩阵 [Total_Edges, 2, d, d]
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        maps = self.map_generator(edge_features).view(-1, 2, self.d, self.d)
        
        # 应用正交约束：Exp(A - A^T)
        # 这确保了扩散过程的数值稳定性，防止过度平滑
        A_u = maps[:, 0] - maps[:, 0].transpose(-1, -2)
        A_v = maps[:, 1] - maps[:, 1].transpose(-1, -2)
        f_u_e = torch.matrix_exp(A_u) # [E, d, d]
        f_v_e = torch.matrix_exp(A_v) # [E, d, d]

        # 开始消息传递计算：L_f * x
        # 传递映射矩阵到 message 函数
        out = self.propagate(edge_index, x=x, f_u_e=f_u_e, f_v_e=f_v_e)
        
        # 严格层扩散方程: x(t+1) = x(t) - eps * L_f * x(t)
        # 这里的 out 实际上是 L_f 对节点的作用结果
        return x - self.eps * out

    def message(self, x_i, x_j, f_u_e, f_v_e):
        """
        x_i, x_j: [E, C] -> [E, k, d]
        f_u_e, f_v_e: [E, d, d]
        """
        # 重塑特征以进行 Stalk-wise 矩阵乘法
        x_i_stalks = x_i.view(-1, self.k, self.d).unsqueeze(-1) # [E, k, d, 1]
        x_j_stalks = x_j.view(-1, self.k, self.d).unsqueeze(-1) # [E, k, d, 1]

        # 核心：计算边上的一致性误差 (F_ue * x_u - F_ve * x_v)
        # 这种表达方式在求和后精确对应了 Sheaf Laplacian 的作用
        proj_i = torch.matmul(f_u_e.unsqueeze(1), x_i_stalks) # [E, k, d, 1]
        proj_j = torch.matmul(f_v_e.unsqueeze(1), x_j_stalks) # [E, k, d, 1]
        
        error = proj_i - proj_j
        
        # 将误差投影回节点空间: F_ue^T * error
        msg = torch.matmul(f_u_e.transpose(-1, -2).unsqueeze(1), error)
        
        return msg.view(-1, self.channels) # [E, C]

    def update(self, aggr_out):
        return self.dropout(aggr_out)

# 示例：支持 Batch 的全连接模型
class SheafGNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, stalk_dim, num_layers=2):
        super().__init__()
        self.lin_in = nn.Linear(in_feat, hidden_feat)
        self.convs = nn.ModuleList([
            StrictBatchSheafConv(hidden_feat, stalk_dim) for _ in range(num_layers)
        ])
        self.lin_out = nn.Linear(hidden_feat, out_feat)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.lin_in(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x) # 论文推荐使用 ELU
            
        return self.lin_out(x)