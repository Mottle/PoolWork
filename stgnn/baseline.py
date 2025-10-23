import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
import torch.nn.functional as F

class BaseLine(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3, backbone = 'gcn'):
        super(BaseLine, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.backbone = backbone
        
        self.build_convs()

    def build_convs(self):
        if self.num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(self.build_conv(self.in_channels, self.hidden_channels))
            else:
                self.convs.append(self.build_conv(self.hidden_channels, self.hidden_channels))

    def build_conv(self, in_channels, out_channels):
        if self.backbone == 'gcn':
            return GCNConv(in_channels, out_channels)
        elif self.backbone == 'gin':
            return GINConv(nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)))
        else:
            raise ValueError("backbone must be 'gcn'")
    
    def forward(self, x, edge_index, batch):
        feature_all = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x)
            feature = global_mean_pool(x, batch)
            feature_all.append(feature)
        merge_feature = torch.mean(torch.stack(feature_all, dim=0), dim=0)

        return merge_feature, 0
