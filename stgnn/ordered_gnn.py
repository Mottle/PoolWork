import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

class OrderedGNNLayer(MessagePassing):
    """
    [Strict Implementation] OrderedGNN Layer
    
    符合论文核心机制：
    1. 学习节点 Rank Score。
    2. 双流聚合 (Dual-Stream Aggregation): 独立聚合 Forward 和 Backward 邻居。
    3. 拼接融合 (Concatenation): 输出为 [Self || Fwd || Bwd]，严格保留方向信息。
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__(aggr='add')  # 聚合本身仍是 Add，但针对不同流分别聚合
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # 1. 排序打分器
        self.score_net = nn.Linear(in_channels, 1)

        # 2. 特征变换 (各自独立)
        # 注意：为了拼接后维度不爆炸，通常会将 out_channels 设为 hidden / 3
        # 或者在这里我们假设输入 out_channels 是最终期望的维度，内部除以 3
        self.lin_fwd = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_bwd = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

        # 3. 最终融合层 (Combine)
        # 将 [Self, Fwd, Bwd] 拼接后映射回 out_channels
        self.lin_combine = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x, edge_index):
        # 1. 计算 Rank Scores
        scores = self.score_net(x)

        # 2. 预计算特征变换
        h_fwd = self.lin_fwd(x)
        h_bwd = self.lin_bwd(x)
        h_self = self.lin_self(x)

        # 3. 消息传递 (同时计算 Fwd 和 Bwd 流)
        # 返回 tuple: (aggr_fwd, aggr_bwd)
        out_fwd, out_bwd = self.propagate(edge_index, x_fwd=h_fwd, x_bwd=h_bwd, scores=scores)

        # 4. 严格拼接 (Concatenation) - 论文核心
        # [N, 3 * out_channels]
        out_concat = torch.cat([h_self, out_fwd, out_bwd], dim=-1)
        
        # 5. 融合输出
        out = self.lin_combine(out_concat)
        
        return out

    def message(self, x_fwd_j, x_bwd_j, scores_i, scores_j):
        """
        同时计算两路消息，返回两个独立的 msg tensor
        """
        # 计算相对顺序权重
        diff = scores_j - scores_i
        alpha_fwd = torch.sigmoid(diff)      # j 是上级的概率
        alpha_bwd = 1.0 - alpha_fwd          # j 是下级的概率
        
        # 显式加权
        msg_fwd = x_fwd_j * alpha_fwd
        msg_bwd = x_bwd_j * alpha_bwd
        
        # Dropout
        msg_fwd = F.dropout(msg_fwd, p=self.dropout, training=self.training)
        msg_bwd = F.dropout(msg_bwd, p=self.dropout, training=self.training)
        
        # 返回 tuple，PyG 的 aggregate 会自动分别处理
        return msg_fwd, msg_bwd

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # inputs 是一个 tuple (msg_fwd, msg_bwd)
        msg_fwd, msg_bwd = inputs
        
        # 分别聚合
        aggr_fwd = super().aggregate(msg_fwd, index, ptr, dim_size)
        aggr_bwd = super().aggregate(msg_bwd, index, ptr, dim_size)
        
        return aggr_fwd, aggr_bwd

class OrderedGNN(nn.Module):
    """
    符合论文架构的完整 OrderedGNN Baseline
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding
        self.node_emb = nn.Linear(in_channels, hidden_channels)

        # Stack OrderedGNN Layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(OrderedGNNLayer(hidden_channels, hidden_channels, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, batch, *args, **kwargs):
        # 1. Embedding
        x = self.node_emb(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Ordered Message Passing
        for i in range(self.num_layers):
            identity = x
            
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x) # LayerNorm 对 OrderedGNN 很重要
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual
            x = x + identity

        # 3. Global Mean Pooling (按要求)
        out = global_mean_pool(x, batch)

        
        return out, 0