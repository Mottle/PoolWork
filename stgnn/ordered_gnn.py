# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing, global_mean_pool

# class OrderedGNNLayer(MessagePassing):
#     """
#     [Strict Implementation] OrderedGNN Layer
    
#     符合论文核心机制：
#     1. 学习节点 Rank Score。
#     2. 双流聚合 (Dual-Stream Aggregation): 独立聚合 Forward 和 Backward 邻居。
#     3. 拼接融合 (Concatenation): 输出为 [Self || Fwd || Bwd]，严格保留方向信息。
#     """
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super().__init__(aggr='add')  # 聚合本身仍是 Add，但针对不同流分别聚合
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dropout = dropout

#         # 1. 排序打分器
#         self.score_net = nn.Linear(in_channels, 1)

#         # 2. 特征变换 (各自独立)
#         # 注意：为了拼接后维度不爆炸，通常会将 out_channels 设为 hidden / 3
#         # 或者在这里我们假设输入 out_channels 是最终期望的维度，内部除以 3
#         self.lin_fwd = nn.Linear(in_channels, out_channels, bias=False)
#         self.lin_bwd = nn.Linear(in_channels, out_channels, bias=False)
#         self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

#         # 3. 最终融合层 (Combine)
#         # 将 [Self, Fwd, Bwd] 拼接后映射回 out_channels
#         self.lin_combine = nn.Linear(out_channels * 3, out_channels)

#     def forward(self, x, edge_index):
#         # 1. 计算 Rank Scores
#         scores = self.score_net(x)

#         # 2. 预计算特征变换
#         h_fwd = self.lin_fwd(x)
#         h_bwd = self.lin_bwd(x)
#         h_self = self.lin_self(x)

#         # 3. 消息传递 (同时计算 Fwd 和 Bwd 流)
#         # 返回 tuple: (aggr_fwd, aggr_bwd)
#         out_fwd, out_bwd = self.propagate(edge_index, x_fwd=h_fwd, x_bwd=h_bwd, scores=scores)

#         # 4. 严格拼接 (Concatenation) - 论文核心
#         # [N, 3 * out_channels]
#         out_concat = torch.cat([h_self, out_fwd, out_bwd], dim=-1)
        
#         # 5. 融合输出
#         out = self.lin_combine(out_concat)
        
#         return out

#     def message(self, x_fwd_j, x_bwd_j, scores_i, scores_j):
#         """
#         同时计算两路消息，返回两个独立的 msg tensor
#         """
#         # 计算相对顺序权重
#         diff = scores_j - scores_i
#         alpha_fwd = torch.sigmoid(diff)      # j 是上级的概率
#         alpha_bwd = 1.0 - alpha_fwd          # j 是下级的概率
        
#         # 显式加权
#         msg_fwd = x_fwd_j * alpha_fwd
#         msg_bwd = x_bwd_j * alpha_bwd
        
#         # Dropout
#         msg_fwd = F.dropout(msg_fwd, p=self.dropout, training=self.training)
#         msg_bwd = F.dropout(msg_bwd, p=self.dropout, training=self.training)
        
#         # 返回 tuple，PyG 的 aggregate 会自动分别处理
#         return msg_fwd, msg_bwd

#     def aggregate(self, inputs, index, ptr=None, dim_size=None):
#         # inputs 是一个 tuple (msg_fwd, msg_bwd)
#         msg_fwd, msg_bwd = inputs
        
#         # 分别聚合
#         aggr_fwd = super().aggregate(msg_fwd, index, ptr, dim_size)
#         aggr_bwd = super().aggregate(msg_bwd, index, ptr, dim_size)
        
#         return aggr_fwd, aggr_bwd

# class OrderedGNN(nn.Module):
#     """
#     符合论文架构的完整 OrderedGNN Baseline
#     """
#     def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5):
#         super().__init__()
#         self.num_layers = num_layers
#         self.dropout = dropout

#         # Embedding
#         self.node_emb = nn.Linear(in_channels, hidden_channels)

#         # Stack OrderedGNN Layers
#         self.layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
        
#         for _ in range(num_layers):
#             self.layers.append(OrderedGNNLayer(hidden_channels, hidden_channels, dropout=dropout))
#             self.norms.append(nn.LayerNorm(hidden_channels))
        
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x, edge_index, batch, *args, **kwargs):
#         # 1. Embedding
#         x = self.node_emb(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # 2. Ordered Message Passing
#         for i in range(self.num_layers):
#             identity = x
            
#             x = self.layers[i](x, edge_index)
#             x = self.norms[i](x) # LayerNorm 对 OrderedGNN 很重要
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
            
#             # Residual
#             x = x + identity

#         # 3. Global Mean Pooling (按要求)
#         out = global_mean_pool(x, batch)

        
#         return out, 0



import torch
import torch.nn.functional as F
# from mp_deterministic import MessagePassing
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, LayerNorm
from torch_geometric.nn import global_mean_pool

# https://github.com/LUMIA-Group/OrderedGNN/blob/main/layer.py
class ONGNNConv(MessagePassing):
    def __init__(self, tm_net, tm_norm, params):
        super(ONGNNConv, self).__init__('mean')
        self.params = params
        self.tm_net = tm_net
        self.tm_norm = tm_norm

    def forward(self, x, edge_index, last_tm_signal):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if self.params['add_self_loops']==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if self.params['add_self_loops']==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        if self.params['tm']==True:
            if self.params['simple_gating']==True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))    
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.params['diff_or']==True:
                    tm_signal_raw = last_tm_signal+(1-last_tm_signal)*tm_signal_raw
            tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)
            out = x*tm_signal + m*(1-tm_signal)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)

        return out, tm_signal_raw
    

# https://github.com/LUMIA-Group/OrderedGNN/blob/main/model.py
# class OrderedGNN(Module):
#     def __init__(self, params):
#         super().__init__()
#         self.params = params
#         self.linear_trans_in = ModuleList()
#         self.linear_trans_out = Linear(params['hidden_channel'], params['out_channel'])
#         self.norm_input = ModuleList()
#         self.convs = ModuleList()

#         self.tm_norm = ModuleList()
#         self.tm_net = ModuleList()

#         self.linear_trans_in.append(Linear(params['in_channel'], params['hidden_channel']))

#         self.norm_input.append(LayerNorm(params['hidden_channel']))

#         for i in range(params['num_layers_input']-1):
#             self.linear_trans_in.append(Linear(params['hidden_channel'], params['hidden_channel']))
#             self.norm_input.append(LayerNorm(params['hidden_channel']))

#         if params['global_gating']==True:
#             tm_net = Linear(2*params['hidden_channel'], params['chunk_size'])

#         for i in range(params['num_layers']):
#             self.tm_norm.append(LayerNorm(params['hidden_channel']))
            
#             if params['global_gating']==False:
#                 self.tm_net.append(Linear(2*params['hidden_channel'], params['chunk_size']))
#             else:
#                 self.tm_net.append(tm_net)
            
#             if params['model']=="ONGNN":
#                 self.convs.append(ONGNNConv(tm_net=self.tm_net[i], tm_norm=self.tm_norm[i], params=params))

#         self.params_conv = list(set(list(self.convs.parameters())+list(self.tm_net.parameters())))
#         self.params_others = list(self.linear_trans_in.parameters())+list(self.linear_trans_out.parameters())

#     def forward(self, x, edge_index, batch, *args, **kwargs):
#         check_signal = []

#         for i in range(len(self.linear_trans_in)):
#             x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
#             x = F.relu(self.linear_trans_in[i](x))
#             x = self.norm_input[i](x)

#         tm_signal = x.new_zeros(self.params['chunk_size'])

#         for j in range(len(self.convs)):
#             if self.params['dropout_rate2']!='None':
#                 x = F.dropout(x, p=self.params['dropout_rate2'], training=self.training)
#             else:
#                 x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
#             x, tm_signal = self.convs[j](x, edge_index, last_tm_signal=tm_signal)
#             check_signal.append(dict(zip(['tm_signal'], [tm_signal])))

#         x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
#         x = self.linear_trans_out(x)

#         encode_values = dict(zip(['x', 'check_signal'], [x, check_signal]))
        
#         return encode_values

class OrderedGNN(Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 num_layers=3, 
                 chunk_size=32, 
                 num_layers_input=1,
                 dropout=0.5, 
                 global_gating=True,
                 add_self_loops=True,
                 simple_gating=False, # 默认为 False 以启用 Ordered 逻辑
                 diff_or=True):
        """
        OrderedGNN 编码器 (Adapted for PyG Graph Classification)
        
        Args:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层维度
            num_layers (int): GNN 层数 (卷积层数)
            chunk_size (int): OrderedGNN 特有的分块大小 (用于 tm_signal)
            num_layers_input (int): 输入 MLP 的层数 (默认 1)
            dropout (float): Dropout 比率
            global_gating (bool): 是否使用全局门控
        """
        super(OrderedGNN, self).__init__()

        # 1. 保存参数供 forward 使用
        self.dropout = dropout
        self.chunk_size = chunk_size
        
        # 为了兼容 ONGNNConv (它可能依赖 params 字典)，我们构建一个内部 params
        self._internal_params = {
            'hidden_channel': hidden_channels,
            'chunk_size': chunk_size,
            'dropout_rate': dropout,
            'add_self_loops': add_self_loops,
            'tm': True,              # 既然叫 OrderedGNN，默认开启 TM 机制
            'simple_gating': simple_gating,
            'diff_or': diff_or,
            'model': 'ONGNN'
        }

        self.linear_trans_in = ModuleList()
        self.norm_input = ModuleList()

        self.linear_trans_in.append(Linear(in_channels, hidden_channels))
        self.norm_input.append(LayerNorm(hidden_channels))

        for i in range(num_layers_input - 1):
            self.linear_trans_in.append(Linear(hidden_channels, hidden_channels))
            self.norm_input.append(LayerNorm(hidden_channels))

        self.convs = ModuleList()
        self.tm_norm = ModuleList()
        self.tm_net = ModuleList()

        shared_tm_net = None
        if global_gating:
            shared_tm_net = Linear(2 * hidden_channels, chunk_size)

        for i in range(num_layers):
            self.tm_norm.append(LayerNorm(hidden_channels))
            
            if global_gating:
                current_tm_net = shared_tm_net
            else:
                current_tm_net = Linear(2 * hidden_channels, chunk_size)
            
            self.tm_net.append(current_tm_net)
            self.convs.append(
                ONGNNConv(tm_net=current_tm_net, 
                          tm_norm=self.tm_norm[i], 
                          params=self._internal_params)
            )

        # 记录输出维度供外部分类器使用
        self.out_dim = hidden_channels

    def forward(self, x, edge_index, batch, *args, **kwargs) -> torch.Tensor:
        
        for i in range(len(self.linear_trans_in)):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.linear_trans_in[i](x)
            x = F.relu(x) # 原代码是在 linear 之后 relu
            x = self.norm_input[i](x)

        tm_signal = x.new_zeros(x.size(0), self.chunk_size)

        # 3. 卷积循环
        for j in range(len(self.convs)):
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 执行卷积，更新 x 和 tm_signal
            # ONGNNConv 通常返回 (x, new_tm_signal)
            x, tm_signal = self.convs[j](x, edge_index, last_tm_signal=tm_signal)

        x = F.dropout(x, p=self.dropout, training=self.training)

        out = global_mean_pool(x, batch)
        
        return out, 0