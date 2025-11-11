import torch
import context
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.nn.norm import GraphNorm
from utils.dirichlet_energy import compute_dirichlet_energy
from utils.mean_average_distance import compute_mean_average_distance

class BaseLine(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3, backbone = 'gcn', dropout = 0.5):
        super(BaseLine, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.backbone = backbone
        self.dropout = 0.5

        if self.in_channels < 1:
            self.in_channels = 1
        
        self._build_embedding()
        self._build_convs()
        self._build_norms()

    def _build_embedding(self):
        self.embedding = nn.Linear(self.in_channels, self.hidden_channels)

    def _build_convs(self):
        if self.num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(self._build_conv(self.hidden_channels, self.hidden_channels))

    def _build_conv(self, in_channels, out_channels):
        if self.backbone == 'gcn':
            return GCNConv(in_channels, out_channels)
        elif self.backbone == 'gin':
            fnn = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(p=self.dropout)
            )
            return GINConv(fnn)
        else:
            raise ValueError("backbone must be 'gcn'")
        
    def _build_norms(self):
        self.norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.norms.append(GraphNorm(self.hidden_channels))
    
    def forward(self, x, edge_index, batch):
        ori_x = x
        x = self.embedding(x)

        feature_all = []
        self.dirichlet_energies = [compute_dirichlet_energy(x, edge_index, batch)]
        self.mads = [compute_mean_average_distance(x, batch)]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            
            self.dirichlet_energies.append(compute_dirichlet_energy(x, edge_index, batch))
            self.mads.append(compute_mean_average_distance(x, batch))

            feature = global_mean_pool(x, batch)
            feature_all.append(feature)
        merge_feature = torch.mean(torch.stack(feature_all, dim=0), dim=0)

        self.dirichlet_energy_rates = []
        self.mad_rates = []
        for i in range(self.num_layers):
            rate = self.dirichlet_energies[i + 1] / self.dirichlet_energies[0]
            self.dirichlet_energy_rates.append(rate)
            mag_rate = self.mads[i + 1] / self.mads[0]
            self.mad_rates.append(mag_rate)
            # print(f'rate: {rate}')
        # average = self.sum(dirichlet_energy_rates) / self.num_layers
        # print(f'average rate: {average}')

        return merge_feature, 0
