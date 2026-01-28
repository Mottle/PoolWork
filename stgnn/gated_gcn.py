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
        super(GatedGCN, self).__init__()
        self.dropout_rate = dropout
        self.num_layers = num_layers

        # 1. 节点特征编码器 (将输入映射到隐藏空间)
        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(ResGatedGraphConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))


    def forward(self, x, edge_index, batch, edge_attr=None, *args, **kwargs):
        
        x = self.node_encoder(x)
        
        for i in range(self.num_layers):
            x_in = x
            x = self.convs[i](x, edge_index, edge_attr)
            # x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + x_in

        global_feature = global_mean_pool(x, batch)
        
        return global_feature, 0
