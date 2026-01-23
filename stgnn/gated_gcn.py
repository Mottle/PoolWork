import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ResGatedGraphConv, global_mean_pool

class GatedGCN(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        # out_channels: int, 
        num_layers: int, 
        dropout: float = 0.5
    ):
        """
        Args:
            in_channels: 节点输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出维度 (例如分类数)
            num_layers: GatedGCN 层数 (n层)
            dropout: Dropout 概率
        """
        super(GatedGCN, self).__init__()
        self.dropout_rate = dropout
        self.num_layers = num_layers

        # 1. 节点特征编码器 (将输入映射到隐藏空间)
        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        # 2. 定义 N 层 GatedGCN 和 BatchNorm
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            # ResGatedGraphConv: 基于残差的门控图卷积
            self.convs.append(ResGatedGraphConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))


    def forward(self, x, edge_index, batch, edge_attr=None, *args, **kwargs):
        """
        Args:
            x: 节点特征 [Num_Nodes, In_Channels]
            edge_index: 边索引 [2, Num_Edges]
            batch: 批次向量 [Num_Nodes], 用于标识节点属于哪个图
        """
        
        # --- A. 初始编码 ---
        x = self.node_encoder(x)
        
        # --- B. N 层消息传递 ---
        for i in range(self.num_layers):
            x_in = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + x_in

        global_feature = global_mean_pool(x, batch)
        
        return global_feature, 0
