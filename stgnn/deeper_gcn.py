import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer

class DeeperGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=28, dropout=0.5):
        super(DeeperGCN, self).__init__()

        self.dropout = dropout
        
        # 1. 节点特征编码器 (Node Encoder)
        # 将原始特征映射到隐藏层维度
        self.node_encoder = Linear(in_channels, hidden_channels)

        # 2. 核心深层 GCN 模块
        self.layers = torch.nn.ModuleList()
        
        for i in range(num_layers):
            # 定义卷积算子 GENConv
            # aggr='softmax': 使用 SoftMax 聚合
            # learn_t=True: 学习温度参数 beta
            # learn_p=True: 如果使用 powermean，可以开启此项学习 p
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')

            # 定义归一化层，这里使用 LayerNorm，也可以配合 MsgNorm 使用
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            
            # 激活函数
            act = ReLU(inplace=True)

            # 使用 DeepGCNLayer 进行包装
            # block='res+': 使用 ResGCN+ 的残差连接方式 (DeeperGCN 推荐)
            # dropout: 在层间添加 dropout
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout, ckpt_grad=False)
            
            self.layers.append(layer)

        # 3. 输出头 (Output Head)
        self.output_head = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 编码输入特征
        x = self.node_encoder(x)
        
        # 逐层通过 DeeperGCN Layer
        # DeepGCNLayer 会自动处理 x = conv(norm(x)) + x 这种 Res+ 的逻辑
        for layer in self.layers:
            x = layer(x, edge_index)

        # 最终输出层
        x = self.output_head(x)
        
        # 这里的输出通常是 Logits，外部可以使用 CrossEntropyLoss 或 F.log_softmax
        return x