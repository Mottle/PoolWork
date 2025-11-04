import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_max_pool, global_add_pool, GraphNorm, knn_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_geometric.data import Data
import networkx as nx
from perf_counter import get_time_sync


class DualRoadGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5, k = 3):
        super(DualRoadGNN, self).__init__()
        self.in_channels = max(in_channels, 1)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k

        if num_layers < 2:
            raise ValueError("Number of layers should be greater than 1.")
        
        if k <= 1:
            raise ValueError("k should be greater than 1.")
        
        self._build_embedding()
        self.convs = self._build_convs()
        self.norms = self._build_graph_norms()
        self.feature_convs = self._build_convs()
        self.feature_norms = self._build_graph_norms()
    
    def _build_embedding(self):
        # self.embedding = nn.Embedding(num_embeddings=self.in_channels, embedding_dim=self.hidden_channels)
        self.embedding = nn.Linear(in_features=self.in_channels, out_features=self.hidden_channels)

    def _build_convs(self):
        convs = nn.ModuleList()
        for i in range(self.num_layers):
            convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
        return convs

    def _build_graph_norms(self):
        graph_norms = nn.ModuleList()
        for i in range(self.num_layers):
            graph_norms.append(GraphNorm(self.hidden_channels))
        return graph_norms

    def forward(self, x, edge_index, batch):
        originl_x = x
        x = self.embedding(x)
        feature_graph_edge_index = knn_graph(x, self.k, batch, loop=False, cosine=True)

        all_x = []

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            feature_x = self.feature_convs[i](x, feature_graph_edge_index)
            feature_x = self.feature_norms[i](feature_x, batch)
            feature_x = F.leaky_relu(feature_x)
            feature_x = F.dropout(feature_x, p=self.dropout, training=self.training)

            fusion_x = (x + feature_x) / 2
            all_x.append(fusion_x)
            x = fusion_x

        graph_feature = 0
        for i in range(self.num_layers):
            graph_feature += global_mean_pool(all_x[i - 1], batch)
        return graph_feature, 0