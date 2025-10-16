import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.pool import TopKPooling, ASAPooling, EdgePooling, SAGPooling, PANPooling
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch import nn
from torch_sparse import SparseTensor
from struct_pooling import StructPooling
from mincut_pooling import MinCutPooling

activate = F.leaky_relu

class Pooler(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim = None, num_layers = 3, pool_type = "topk", gnn_type = "gcn", skip_link = True):
        super().__init__()
        self.pool_type = pool_type
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.skip_link = skip_link

        if out_dim == None:
            out_dim = hidden_dim

        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2")
        
        if skip_link:
            self.real_out_dim = out_dim * num_layers * 2
        else:
            self.real_out_dim = out_dim * 2

        #用于某些没有x的数据集
        if self.in_dim <= 0:
            self.in_dim = 1
        
        self.build_model()

    def build_model(self):
        self.conv = torch.nn.ModuleList()
        self.pool = torch.nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                in_d = self.in_dim
                out_d = self.hidden_dim
            # if i == self.num_layers - 1:
            #     in_d = self.hidden_dim
            #     out_d = self.out_dim
            else:
                in_d = self.hidden_dim
                out_d = self.hidden_dim

            if self.gnn_type == "gcn":
                self.conv.append(GCNConv(in_d, out_d))
            elif self.gnn_type == "gin":
                self.conv.append(GINConv(nn.Linear(in_d, out_d)))
            else:
                raise ValueError(f'Invalid GNN type: {self.gnn_type}')

        for i in range(self.num_layers):
            in_d = self.hidden_dim
            # out_d = self.hidden_dim

            if self.pool_type.lower() == 'topk':
                self.pool.append(TopKPooling(in_channels=in_d))
            # elif self.pool_type.lower() == 'mean':
            #     self.pool.append(torch.nn.AvgPool1d(kernel_size=1))
            # elif self.pool_type.lower() == 'max':
            #     self.pool.append(torch.nn.MaxPool1d(kernel_size=1))
            elif self.pool_type.lower() == 'asap':
                self.pool.append(ASAPooling(in_channels=in_d))
            elif self.pool_type.lower() == 'edge':
                self.pool.append(EdgePooling(in_channels=in_d))
            elif self.pool_type.lower() == 'sag':
                self.pool.append(SAGPooling(in_channels=in_d))
            elif self.pool_type.lower() == 'pan':
                self.pool.append(PANPooling(in_channels=in_d))
            elif self.pool_type.lower() == 'struct':
                self.pool.append(StructPooling(in_channels=in_d))
            elif self.pool_type.lower() == 'mincut':
                self.pool.append(MinCutPooling(in_channels=in_d))
            else:
                raise ValueError(f'Invalid pool type: {self.pool_type}')

    def get_out_dim(self):
        return self.real_out_dim

    def forward(self, x, edge_index, batch = None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 存储各层特征
        layer_features = []
        original_x = x

        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = activate(x)
            
            if self.pool_type.lower() == 'topk':
                x, edge_index, _, batch, _, _ = self.pool[i](x, edge_index, batch=batch)
            elif self.pool_type.lower() == 'asap':
                x, edge_index, _, batch, _ = self.pool[i](x, edge_index, batch=batch)
            elif self.pool_type.lower() == 'edge':
                x, edge_index, batch, _ = self.pool[i](x, edge_index, batch=batch)
            elif self.pool_type.lower() == 'sag':
                x, edge_index, _, batch, _, _ = self.pool[i](x, edge_index, batch=batch)
            elif self.pool_type.lower() == 'pan':
                row, col = edge_index
                edge_attr = torch.ones(row.size(0))
                sparse_adj = SparseTensor(row=row.to(x.device), col=col.to(x.device), value=edge_attr.to(x.device), sparse_sizes=(x.size(0), x.size(0))).to_device(x.device)
                x, edge_index, _, batch, _, _ = self.pool[i](x, sparse_adj, batch=batch)
            elif self.pool_type.lower() == 'struct':
                x, edge_index, batch = self.pool[i](x, edge_index, batch=batch)
            
            global_feat = (global_max_pool(x, batch), global_mean_pool(x, batch))
            global_feat = torch.cat([global_feat[0], global_feat[1]], dim=1)
            layer_features.append(global_feat)
        
        x = torch.cat(layer_features, dim=1)

        return x
    
class Classifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.liner0 = torch.nn.Linear(in_dim, hidden_dim)
        self.liner1 = torch.nn.Linear(hidden_dim, num_classes)

        # self._initialize_weights()
    
    def _initialize_weights(self):
        # Kaiming/He 初始化（适合ReLU族激活函数）
        torch.nn.init.kaiming_uniform_(self.liner0.weight)
        torch.nn.init.kaiming_uniform_(self.liner1.weight)
    
        # 初始化偏置为0
        if self.liner0.bias is not None:
            nn.init.constant_(self.liner0.bias, 0)
        if self.liner1.bias is not None:
            nn.init.constant_(self.liner1.bias, 0)

    def forward(self, x):
        x = activate(self.liner0(x))
        x = self.liner0(x)
        # x = activate(x)
        x = F.leaky_relu(x)
        x = torch.dropout(x, p=0.4, train=self.training)
        x = self.liner1(x)

        return x