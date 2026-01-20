import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add


class DirectionalConv(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_weight, deg_inv=None):
        row, col = edge_index
        msg = x[col] * edge_weight.unsqueeze(-1)
        out = scatter_add(msg, row, dim=0, dim_size=x.size(0))
        if deg_inv is not None:
            out = out * deg_inv.unsqueeze(-1)
        return out


class DGN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=4, dropout=0.5, k_dirs=1):
        super().__init__()
        self.dropout = dropout
        self.k_dirs = k_dirs
        self.hidden_channels = hidden_channels

        self.node_encoder = Linear(in_channels, hidden_channels)
        self.conv = DirectionalConv()

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            num_channels = 1 + 1 + 2 * k_dirs  # self + avg + dir+ + dir-
            fusion_lin = Linear(hidden_channels * num_channels, hidden_channels)
            self.layers.append(fusion_lin)

    def compute_direction_weights(self, edge_index, pe):
        row, col = edge_index
        u_i = pe[row]   # [E, k_dirs]
        u_j = pe[col]   # [E, k_dirs]
        diff = u_j - u_i
        w_up = F.leaky_relu(diff)
        w_down = F.leaky_relu(-diff)
        return w_up, w_down

    def compute_degree_norm(self, edge_index, num_nodes):
        row, col = edge_index
        deg = scatter_add(torch.ones(row.size(0), device=row.device),
                          row, dim=0, dim_size=num_nodes)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv

    def forward(self, x, edge_index, batch, pe, *args, **kwargs):
        x = self.node_encoder(x)

        deg_inv = self.compute_degree_norm(edge_index, x.size(0))
        w_up, w_down = self.compute_direction_weights(edge_index, pe)
        w_avg = torch.ones(edge_index.size(1), device=x.device)

        for fusion_lin in self.layers:
            aggregations = []

            aggregations.append(x)  # self
            out_avg = self.conv(x, edge_index, w_avg, deg_inv)
            aggregations.append(out_avg)

            for d in range(self.k_dirs):
                out_up = self.conv(x, edge_index, w_up[:, d], deg_inv)
                out_down = self.conv(x, edge_index, w_down[:, d], deg_inv)
                aggregations.append(out_up)
                aggregations.append(out_down)

            x_concat = torch.cat(aggregations, dim=-1)
            x_new = fusion_lin(x_concat)

            x = x + x_new
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return x, 0


# import torch
# import torch.nn.functional as F
# from torch.nn import Linear, ReLU, LayerNorm
# from torch_geometric.nn import GCNConv, global_mean_pool
# import torch_geometric.transforms as T
# from torch_geometric.data import Data, Batch

# class DGN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers=4, dropout=0.5):
#         super(DGN, self).__init__()
#         self.dropout = dropout

#         self.node_encoder = Linear(in_channels, hidden_channels)
#         self.layers = torch.nn.ModuleList()
        
#         for i in range(num_layers):
#             conv = GCNConv(hidden_channels, hidden_channels)
#             self.layers.append(conv)


#     def get_directional_edge_weights(self, edge_index, eig_vecs):
#         row, col = edge_index
        
#         u_i = eig_vecs[row, 1] 
#         u_j = eig_vecs[col, 1]
        
#         # 计算梯度绝对值
#         # 注意：这里加一个 1e-5 或者 +1 可以避免权重为0导致断连，
#         # 但DGN原意就是让梯度大的地方权重更大，所以直接用 abs 也可以。
#         edge_weight = torch.abs(u_i - u_j)
        
#         return edge_weight

#     def forward(self, x, edge_index, batch, pe, *args, **kwargs):
        
#         edge_weight = self.get_directional_edge_weights(edge_index, pe)

#         x = self.node_encoder(x)

#         for layer in self.layers:
#             x = layer(x, edge_index, edge_weight=edge_weight)
            
#             x = F.leaky_relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         x = global_mean_pool(x, batch)
        
#         return x, 0