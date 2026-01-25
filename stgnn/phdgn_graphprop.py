import torch
from phdgn_utils import PortHamiltonianConv
import torch
from typing import Optional
from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, GraphNorm
from collections import OrderedDict

# https://github.com/simonheilig/porthamiltonian-dgn/blob/main/graph_prop_pred/models/phdgn_graphprop.py

# class PHDGN_GraphProp(Module):
#     def __init__(self, 
#                  input_dim,
#                  output_dim,
#                  hidden_dim,
#                  num_layers,
#                  epsilon,
#                  activ_fun='tanh',
#                  p_conv_mode: str = 'naive',
#                  q_conv_mode: str = 'naive',
#                  doubled_dim: bool = True,
#                  final_state: str = 'pq',
#                  alpha: float = 0.,
#                  beta: float = 0.,
#                  dampening_mode: Optional[str] = None,
#                  external_mode : Optional[str] = None,
#                  dtype=torch.float32,
#                  node_level_task=False,
#                  train_weights: bool = True, 
#                  weight_sharing: bool = True,
#                  bias: bool = True) -> None:
        
#         super().__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.epsilon = epsilon
#         self.activ_fun = activ_fun
#         self.p_conv_mode = p_conv_mode
#         self.q_conv_mode = q_conv_mode
#         self.doubled_dim = doubled_dim
#         self.final_state = final_state
#         self.alpha = alpha
#         self.beta = beta
#         self.dampening_mode = dampening_mode
#         self.external_mode = external_mode
#         self.dtype = dtype
#         self.node_level_task = node_level_task
#         self.train_weights = train_weights
#         self.weight_sharing = weight_sharing
#         self.bias = bias

#         self.emb = Linear(self.input_dim, self.hidden_dim)
#         self.nhid = self.hidden_dim*2 if self.doubled_dim else self.hidden_dim

#         self.convs = ModuleList()
#         for _ in range(1 if self.weight_sharing else self.num_layers):
#             self.convs.append(PortHamiltonianConv(
#                 in_channels=self.nhid,
#                 num_iters=self.num_layers if self.weight_sharing else 1, 
#                 epsilon=epsilon,
#                 activ_fun=activ_fun,
#                 p_conv_mode=p_conv_mode, 
#                 q_conv_mode=q_conv_mode,bias=bias, 
#                 beta=beta, 
#                 alpha=alpha, 
#                 dampening_mode=dampening_mode, 
#                 external_mode=external_mode, 
#                 dtype=dtype
#             ))
            
#         if self.final_state != 'pq':
#             self.nhid = self.nhid // 2

#         if not train_weights:
#             #for param in self.enc.parameters():
#             #    param.requires_grad = False
#             for param in self.conv.parameters():
#                 param.requires_grad = False

#         self.node_level_task = node_level_task 
#         if self.node_level_task:
#             self.readout = Sequential(OrderedDict([
#                 ('L1', Linear(self.nhid, self.nhid // 2)),
#                 ('LeakyReLU1', LeakyReLU()),
#                 ('L2', Linear(self.nhid // 2, self.output_dim)),
#                 ('LeakyReLU2', LeakyReLU())
#             ]))
#         else:
#             self.readout = Sequential(OrderedDict([
#                 ('L1', Linear(self.nhid * 3, (self.nhid * 3) // 2)),
#                 ('LeakyReLU1', LeakyReLU()),
#                 ('L2', Linear((self.nhid * 3) // 2, self.output_dim)),
#                 ('LeakyReLU2', LeakyReLU())
#             ]))

#     def forward(self, data) -> torch.Tensor:
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         h = self.emb(x)
        
#         if self.doubled_dim:
#             h = torch.cat([h,h],dim=1)

#         for conv in self.convs:
#             h = conv(h, edge_index)
        
#         if self.final_state == 'p': # taking p
#             h = h[:,:self.nhid]
#         elif self.final_state == 'q': # taking q
#             h = h[:,self.nhid:]
#         else: # self.final_state == 'pq'
#             pass # x contains both p and q already

#         if not self.node_level_task:
#             h = torch.cat([global_add_pool(h, batch), global_max_pool(h, batch), global_mean_pool(h, batch)], dim=1)

#         h = self.readout(h)
#         return h

class PHDGN(Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 num_layers,
                 epsilon,
                 activ_fun='tanh',
                 p_conv_mode: str = 'naive',
                 q_conv_mode: str = 'naive',
                 doubled_dim: bool = False,
                 final_state: str = 'pq',
                 alpha: float = 0.,
                 beta: float = 0.,
                 dampening_mode=None,
                 external_mode=None,
                 dtype=torch.float32,
                 train_weights: bool = True, 
                 weight_sharing: bool = True,
                 bias: bool = True,
                 use_graph_norm: bool = False, #modified
                 **kwargs
                 ) -> None:
        
        super().__init__()

        self.input_dim = input_dim
        # self.output_dim 被移除，因为不包含分类头
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.activ_fun = activ_fun
        self.p_conv_mode = p_conv_mode
        self.q_conv_mode = q_conv_mode
        self.doubled_dim = doubled_dim
        self.final_state = final_state
        self.alpha = alpha
        self.beta = beta
        self.dampening_mode = dampening_mode
        self.external_mode = external_mode
        self.dtype = dtype
        self.train_weights = train_weights
        self.weight_sharing = weight_sharing
        self.bias = bias
        self.use_graph_norm = use_graph_norm  # modified

        self.emb = Linear(self.input_dim, self.hidden_dim)
        
        # 确定隐藏层维度
        # 如果 doubled_dim=True (默认), 内部状态是 [p, q] 拼接，维度是 hidden_dim * 2
        self.nhid = self.hidden_dim * 2 if self.doubled_dim else self.hidden_dim

        self.convs = ModuleList()
        num_conv_instances = 1 if self.weight_sharing else self.num_layers
        
        self.graph_norms = ModuleList()
        for _ in range(num_conv_instances):
            self.convs.append(PortHamiltonianConv(
                in_channels=self.nhid,
                num_iters=self.num_layers if self.weight_sharing else 1, 
                epsilon=epsilon,
                activ_fun=activ_fun,
                p_conv_mode=p_conv_mode, 
                q_conv_mode=q_conv_mode,
                bias=bias, 
                beta=beta, 
                alpha=alpha, 
                dampening_mode=dampening_mode, 
                external_mode=external_mode, 
                dtype=dtype
            ))
            if use_graph_norm:
                self.graph_norms.append(GraphNorm(self.nhid))
            
        if not train_weights:
            for param in self.convs.parameters():
                param.requires_grad = False

        # --- 计算最终输出维度并存储，供外部 Classifier 使用 ---
        # 如果 final_state 是 'p' 或 'q'，我们只取一半的维度
        if self.final_state in ['p', 'q']:
            self.out_dim = self.nhid // 2
        else: # 'pq'
            self.out_dim = self.nhid

        # self.post_linear = Linear(self.out_dim * 2, self.out_dim)

    def forward(self, x, edge_index, batch, *args, **kwargs) -> torch.Tensor:

        h = self.emb(x)
        
        if self.doubled_dim:
            h = torch.cat([h, h], dim=1)

        for conv in self.convs:
            h = conv(h, edge_index)
            if self.use_graph_norm:
                h = self.graph_norms[0](h, batch)
        
        # 根据 final_state 选择输出 p, q 还是 pq
        # 注意：conv输出的 h 维度始终是 self.nhid (即 hidden*2 如果 doubled)
        if self.doubled_dim:
            half_dim = self.nhid // 2
            if self.final_state == 'p': 
                h = h[:, :half_dim]
            elif self.final_state == 'q': 
                h = h[:, half_dim:]
            else: # 'pq'
                pass 
        
        h = global_mean_pool(h, batch)
        # h = self.post_linear(h)

        return h, 0