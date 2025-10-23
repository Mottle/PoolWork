import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv, GCN2Conv, GINConv

class MamboPoolingWithLineGraph(nn.Module):
    def __init__(self, in_channels, dropout=0.0):
        super(MamboPoolingWithLineGraph, self).__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.line_transformer = LineGraph(force_directed=True)
        self.edge_attr_conv = nn.Linear(3 * self.in_channels, self.in_channels)
        self.line_graph_conv = GCNConv(self.in_channels, self.in_channels, add_self_loops=True)

    # def build_line_graph_conv(self):
    #     self.line_graph_conv = GCNConv(self.in_channels, self.self.in_channels, add_self_loops=True)

    def forward(self, x, edge_index, batch, edge_attr = None):

        if edge_attr is None:
            edge_attr = torch.zero((edge_index.size(1), 1), dtype=torch.float, device=x.device)

        merged_edge_attr = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=1)
        merged_edge_attr = self.edge_attr_conv(merged_edge_attr)
        new_data = Data(x=x, edge_index=edge_index, edge_attr=merged_edge_attr, batch=batch)

        line_data = self.line_transformer(new_data)
        line_x, line_edge_index, line_batch = line_data.x, line_data.edge_index, line_data.batch
        # line_x = F.dropout(line_x, p=self.dropout, training=self.training)
        line_x = self.line_graph_conv(line_x, line_edge_index)
        line_x = F.leaky_relu(line_x)

        

        # Aggregate line graph features