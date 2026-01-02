import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class SheafConv(MessagePassing):
    def __init__(self, in_channels, out_channels, stalk_dim):
        """
        in_channels: 节点的输入特征维度 (C)
        out_channels: 节点的输出特征维度 (C')
        stalk_dim: 每个节点 Stalk 的维度 (d) 
                   注意：in_channels 必须能被 stalk_dim 整除
        """
        super(SheafConv, self).__init__(aggr='add')
        self.d = stalk_dim
        self.C = in_channels
        self.num_stalks = in_channels // stalk_dim
        
        # Sheaf Learner: 给每条边学习一个 d*d 的限制映射
        # 为了简化，我们假设映射是正交的或任意线性的
        self.sheaf_learner = nn.Sequential(
            nn.Linear(2 * in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, self.d * self.d)
        )
        
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x shape: [N, C]
        row, col = edge_index
        
        # 1. 动态生成边上的限制映射 F_{u->e} 和 F_{v->e}
        # 这里简化为针对每条边生成一个映射矩阵
        edge_attr = torch.cat([x[row], x[col]], dim=-1) # [E, 2C]
        maps = self.sheaf_learner(edge_attr).view(-1, self.d, self.d) # [E, d, d]
        
        # 2. 消息传递：计算 L_f * x
        # 我们将 x 视为 [N, num_stalks, d]
        out = self.propagate(edge_index, x=x, maps=maps)
        
        # 3. 更新节点特征
        return self.lin(out)

    def message(self, x_j, maps):
        # x_j: [E, C] -> 邻居的特征
        # maps: [E, d, d] -> 边上的映射
        
        # 将特征重塑为 [E, num_stalks, d] 以便进行矩阵乘法
        x_j_reshaped = x_j.view(-1, self.num_stalks, self.d) # [E, k, d]
        
        # 应用限制映射: F_{j->e} * x_j (在 stalk 维度上并行)
        # 使用 einsum 进行批量矩阵乘法: (E, d, d) x (E, k, d) -> (E, k, d)
        msg = torch.einsum('eij,ekj->eki', maps, x_j_reshaped)
        
        return msg.view(-1, self.C) # [E, C]

    def update(self, aggr_out, x):
        # 计算 Sheaf Laplacian 的扩散: x_new = x - L_f * x
        # 这里的 aggr_out 实际上是邻居经过映射后的加和
        # 真正的 L_f * x 包含对角项和非对角项，此处简化为扩散形式
        return x + aggr_out