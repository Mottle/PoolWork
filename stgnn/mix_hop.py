import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MixHopConv, global_mean_pool

class MixHop(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super(MixHop, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # 1. 强制 Input Embedding
        # 将原始特征映射到隐层维度，确保后续计算维度统一
        self.embedding = nn.Linear(in_channels, hidden_channels)

        # 2. 构建 MixHop 层
        # 使用 ModuleList 存储每一层的组件
        self.convs = nn.ModuleList()
        self.fusions = nn.ModuleList()

        # 定义 MixHop 的阶数：[0, 1, 2] 分别代表 自身、1阶邻居、2阶邻居
        # 这是 MixHop 论文中最标准的设置
        self.powers = [0, 1, 2]
        mixhop_output_dim = hidden_channels * len(self.powers)

        for _ in range(num_layers):
            # MixHopConv: 提取多跳特征
            # out_channels 设置为 hidden_channels，意味着每个 scale 输出 hidden_channels
            # 总输出维度将是: hidden_channels * 3
            self.convs.append(
                MixHopConv(hidden_channels, hidden_channels, powers=self.powers)
            )
            
            # Linear Fusion: 融合多跳特征
            # 将 [Batch, 3 * Hidden] 投影回 [Batch, Hidden]
            # 这一步至关重要：它让模型学习不同跳数的重要性
            self.fusions.append(
                nn.Linear(mixhop_output_dim, hidden_channels)
            )

    def forward(self, x, edge_index, batch, *args, **kwargs):
        # 1. Input Projection
        x = self.embedding(x)
        
        # 2. Layer Loop
        for i in range(self.num_layers):
            # Step A: MixHop 多尺度卷积
            # Output shape: [Num_Nodes, Hidden * 3]
            x = self.convs[i](x, edge_index)
            
            # Step B: 线性融合与降维
            # Output shape: [Num_Nodes, Hidden]
            x = self.fusions[i](x)
            
            # Step C: 非线性激活与正则化
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Readout (Graph Pooling)
        # 将节点特征聚合为图特征
        # Output shape: [Batch_Size, Hidden]
        graph_feature = global_mean_pool(x, batch)

        # 返回图特征，第二个返回值留空（为兼容你之前的接口）
        return graph_feature, 0