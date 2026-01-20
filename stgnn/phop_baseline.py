
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F
from phop import PHopGCNConv, PHopGINConv, PHopLinkRWConv, PHopLinkGCNConv, PHopLinkGINConv
from torch_geometric.nn import GINConv, GCNConv, GATConv
from dual_road_gnn import k_farthest_graph
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse, to_dense_adj


class PHopBaseLine(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3, backbone = 'gcn', dropout = 0.5, embed: bool = True):
        super(PHopBaseLine, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.backbone = backbone
        self.dropout = dropout
        self.embed = embed
        self.need_phop = True

        if self.in_channels < 1:
            self.in_channels = 1

        if self.embed:
            self.embedding = nn.Linear(self.in_channels, hidden_channels)
            self.in_channels = hidden_channels
        
        self.build_convs()
        self.build_norms()

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
        if self.backbone == 'phop_linkgcn':
            return PHopLinkGCNConv(in_channels, out_channels, P=3)
        elif self.backbone == 'phop_gcn':
            return PHopGCNConv(in_channels, out_channels, p=2)
        elif self.backbone == 'phop_gin':
            return PHopGINConv(in_channels, out_channels, p=2)
        else:
            raise ValueError(f"backbone invalid: {self.backbone}")
        
    def build_norms(self):
        self.norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.norms.append(GraphNorm(self.hidden_channels))
    
    def forward(self, x, edge_index, Aphop = None, Aphop_loop = None, batch=None):
        if self.embed:
            x = self.embedding(x)

        feature_all = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, Aphop = Aphop, Aphop_loop = Aphop_loop)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            feature = global_mean_pool(x, batch)
            feature_all.append(feature)
        merge_feature = torch.mean(torch.stack(feature_all, dim=0), dim=0)

        return merge_feature, 0

class HybirdPhopGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5, k = 3, p = 3, backbone = 'gcn', self_loops: bool = True):
        super(HybirdPhopGNN, self).__init__()
        self.in_channels = max(in_channels, 1)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k #k farthest
        self.p = p #p hop
        self.backbone = backbone
        self.self_loop = self_loops
        self.push_pe = None

        if num_layers < 2:
            raise ValueError("Number of layers should be greater than 1.")
        
        if k <= 1:
            raise ValueError("k should be greater than 1.")
        
        self._build_embedding()
        self.convs = self._build_convs()
        self.norms = self._build_graph_norms()
        self.feature_convs = self._build_feature_convs()
        self.feature_norms = self._build_graph_norms()
        self.fusion_gate_linear = nn.Linear(self.hidden_channels * 2, hidden_channels)
    
    def _build_embedding(self):
        # self.embedding = nn.Embedding(num_embeddings=self.in_channels, embedding_dim=self.hidden_channels)
        self.embedding = nn.Linear(in_features=self.in_channels, out_features=self.hidden_channels)

    def _build_convs(self):
        convs = nn.ModuleList()
        for _ in range(self.num_layers):
            if self.backbone == 'gcn':
                convs.append(PHopLinkGCNConv(self.hidden_channels, self.hidden_channels, P = self.p))
            elif self.backbone == 'rw':
                convs.append(PHopLinkRWConv(self.hidden_channels, self.hidden_channels, P = self.p))   
            elif self.backbone == 'gin':
                convs.append(PHopLinkGINConv(self.hidden_channels, self.hidden_channels, P = self.p))
            else:
                raise NotImplementedError
        return convs
    
    def _build_feature_convs(self):
        convs = nn.ModuleList()
        for _ in range(self.num_layers):
            if self.backbone == 'gcn':
                convs.append(PHopLinkGCNConv(self.hidden_channels, self.hidden_channels, P = self.p))
            elif self.backbone == 'rw':
                convs.append(PHopLinkRWConv(self.hidden_channels, self.hidden_channels, P = self.p))
            elif self.backbone == 'gin':
                convs.append(PHopLinkGINConv(self.hidden_channels, self.hidden_channels, P = self.p))
            else:
                raise NotImplementedError
        return convs

    def _build_graph_norms(self):
        graph_norms = nn.ModuleList()
        for i in range(self.num_layers):
            graph_norms.append(GraphNorm(self.hidden_channels))
        return graph_norms
    
    def _build_auxiliary_graph(self, x, batch):
        return k_farthest_graph(x, self.k, batch, loop=True, cosine=True, direction=True)
    
    def process_phop(self, x, edge_index, source = None, mode = 'A'):
        if source is not None:
            if mode == 'A':
                prefix = 'a'
            else:
                prefix = 'u'
            edge_index_l = []
            edge_weight_l = []
            for p in range(self.p):
                edge_index_l.append(source[f'{prefix}_{p}_edge_index'])
                edge_weight_l.append(source[f'{prefix}_{p}_edge_weight'])
            return edge_index_l, edge_weight_l

        N = x.size(0)
        # A = to_dense_adj(edge_index, max_num_nodes=N)[0]  # 稠密邻接矩阵 [N, N]

        if self.self_loop:
            # A = A + torch.eye(N, device=A.device)  # 自环
            edge_index, _ = add_self_loops(edge_index)
        
        # A_phop = []
        # for p in range(1, self.p + 1):
        #     Ap = torch.matrix_power(A, p)
        #     A_phop.append(dense_to_sparse(Ap))
        
        # return A_phop
        if mode == 'A':
            return compute_A_phop(edge_index, N, self.p)
        elif mode == 'U':
            return compute_U_phop(edge_index, N, self.p)

    # @torch.compile
    def forward(self, x, edge_index, batch, source = None, *args, **kwargs):
        originl_x = x
        x = self.embedding(x)
        
        feature_graph_edge_index = self._build_auxiliary_graph(x, batch)
        SMp = self.process_phop(x, edge_index, source, mode='U')
        FMp = self.process_phop(x, feature_graph_edge_index, mode='U')

        all_x = []

        for i in range(self.num_layers):
            prev_x = x

            x = self.convs[i](x, edge_index, A_phop = SMp)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            feature_x = self.feature_convs[i](x, feature_graph_edge_index, A_phop = FMp)
            feature_x = self.feature_norms[i](feature_x, batch)
            feature_x = F.leaky_relu(feature_x)
            feature_x = F.dropout(feature_x, p=self.dropout, training=self.training)

            combined = torch.cat([x, feature_x], dim=-1)
            gate = torch.sigmoid(self.fusion_gate_linear(combined))

            fusion_x = gate * x + (1 - gate) * feature_x + prev_x

            # fusion_x = self.fusion_dense_mlp(combined) + prev_x
            all_x.append(fusion_x)
            x = fusion_x

        # graph_feature = 0
        # for i in range(self.num_layers):
        #     graph_feature += global_mean_pool(all_x[i - 1], batch)
        graph_feature = global_mean_pool(all_x[-1], batch)
        return graph_feature, 0
    
def compute_A_phop(edge_index, num_nodes, P):
    A_sparse = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1), device=edge_index.device),
        (num_nodes, num_nodes)
    ).coalesce()
    Ap = A_sparse
    edge_index_l = []
    edge_weight_l = []
    for _ in range(1, P+1):
        Ap = Ap.coalesce()
        edge_index_p = Ap.indices()
        edge_weight_p = Ap.values()
        # results.append((edge_index_p, edge_weight_p))
        edge_index_l.append(edge_index_p)
        edge_weight_l.append(edge_weight_p)
        Ap = torch.sparse.mm(Ap, A_sparse)
    return edge_index_l, edge_weight_l

def compute_U_phop(edge_index, num_nodes, P):
    A_sparse = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1), device=edge_index.device),
        (num_nodes, num_nodes)
    ).coalesce()
    
    # A^0 = I
    A_prev = torch.sparse_coo_tensor(
        torch.arange(num_nodes, device=edge_index.device).repeat(2,1),
        torch.ones(num_nodes, device=edge_index.device),
        (num_nodes, num_nodes)
    ).coalesce()
    
    Ap = A_sparse
    edge_index_l = []
    edge_weight_l = []
    
    for _ in range(1, P+1):
        # 差分 U_p = A^p - A^(p-1)
        U_p = Ap - A_prev
        U_p = U_p.coalesce()
        
        edge_index_p = U_p.indices()
        edge_weight_p = U_p.values()
        edge_index_l.append(edge_index_p)
        edge_weight_l.append(edge_weight_p)
        
        # 更新 A^(p-1) 和 A^p
        A_prev = Ap
        Ap = torch.sparse.mm(Ap, A_sparse).coalesce()
    
    return edge_index_l, edge_weight_l
