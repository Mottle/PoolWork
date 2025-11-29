import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, GraphNorm, knn_graph
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_geometric.data import Data
import networkx as nx
from perf_counter import get_time_sync
from torch import Tensor
from typing import Optional
from torch_geometric.utils import scatter
from reverse_attention import bidirectional_reverse_self_attention


class DualRoadRevAttnGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5, backbone = 'gcn'):
        super(DualRoadRevAttnGNN, self).__init__()
        self.in_channels = max(in_channels, 1)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = backbone

        if num_layers < 2:
            raise ValueError("Number of layers should be greater than 1.")
        
        self._build_embedding()
        self.convs = self._build_convs()
        self.norms = self._build_graph_norms()
        self.feature_convs = self._build_convs()
        self.feature_norms = self._build_graph_norms()
        self.fusion_gate_linear = nn.Linear(self.hidden_channels * 2, hidden_channels)

        self.q_linear = nn.Linear(hidden_channels, hidden_channels)
        self.k_linear = nn.Linear(hidden_channels, hidden_channels)
        self.v_linear = nn.Linear(hidden_channels, hidden_channels)
        self.o_linear = nn.Linear(hidden_channels, hidden_channels)
    
    def _build_embedding(self):
        # self.embedding = nn.Embedding(num_embeddings=self.in_channels, embedding_dim=self.hidden_channels)
        self.embedding = nn.Linear(in_features=self.in_channels, out_features=self.hidden_channels)

    def _build_convs(self):
        convs = nn.ModuleList()
        for i in range(self.num_layers):
            if self.backbone == 'gcn':
                convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
        return convs

    def _build_graph_norms(self):
        graph_norms = nn.ModuleList()
        for i in range(self.num_layers):
            graph_norms.append(GraphNorm(self.hidden_channels))
        return graph_norms
    
    def _build_auxiliary_graph(self, x, batch):
        feature_graph_edge_index = knn_graph(x, self.k, batch, loop=True, cosine=True)
        return feature_graph_edge_index

    def forward(self, x, edge_index, batch):
        originl_x = x
        x = self.embedding(x)
        
        # feature_graph_edge_index = self._build_auxiliary_graph(x, batch)

        all_x = []  

        for i in range(self.num_layers - 1):
            prev_x = x

            x = self.convs[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # feature_x = self.feature_convs[i](x, feature_graph_edge_index)
            # feature_x = self.feature_norms[i](feature_x, batch)
            # feature_x = F.leaky_relu(feature_x)
            # feature_x = F.dropout(feature_x, p=self.dropout, training=self.training)

            Q = self.q_linear(x)
            K = self.k_linear(x)
            V = self.v_linear(x)
            rev_attn_x, _, _ = bidirectional_reverse_self_attention(Q, K, V, batch)
            rev_attn_x = self.o_linear(rev_attn_x)

            combined = torch.cat([x, rev_attn_x], dim=-1)
            gate = torch.sigmoid(self.fusion_gate_linear(combined))

            fusion_x = gate * x + (1 - gate) * rev_attn_x + prev_x
            all_x.append(fusion_x)
            x = fusion_x

        graph_feature = 0
        for i in range(self.num_layers):
            graph_feature += global_mean_pool(all_x[i - 1], batch)
        return graph_feature, 0
    