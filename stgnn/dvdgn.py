from sympy import fu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import global_mean_pool
from phop_baseline import compute_A_phop, compute_U_phop
from dual_road_gnn import k_farthest_graph


class OPDMPConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        P: int = 3,
        dirs: int = 1,
    ):
        super(OPDMPConv, self).__init__()
        self.P = P
        self.dirs = dirs
        self.out_channels = out_channels

        # Hop attention
        self.d = nn.Parameter(torch.ones(P, out_channels))
        self.hop_bias = nn.Parameter(torch.zeros(P, out_channels))

        # Node encoder
        self.lin_in = nn.Linear(in_channels, out_channels)

        # Fusion layers for each hop
        num_channels = 1 + 1 + 2 * dirs
        fusion_in_dim = out_channels * num_channels

        self.hop_fusions = nn.ModuleList(
            [nn.Linear(fusion_in_dim, out_channels) for _ in range(P)]
        )

    @torch.no_grad()
    def precompute(self, M, pe, N):
        edge_indices_list, edge_weights_list = M

        # 只取前 self.dirs 个方向
        pe = pe[:, :self.dirs]

        deg_inv_list = []
        w_up_list = []
        w_down_list = []

        for p in range(self.P):
            edge_index_p = edge_indices_list[p]
            topo_weight_p = edge_weights_list[p]
            row, col = edge_index_p

            deg = scatter_add(topo_weight_p, row, dim=0, dim_size=N)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float("inf")] = 0
            deg_inv_list.append(deg_inv)

            u_i = pe[row]   # [E, self.dirs]
            u_j = pe[col]
            diff = u_j - u_i
            w_up = F.relu(diff)
            w_down = F.relu(-diff)

            w_up_list.append(w_up)
            w_down_list.append(w_down)

        return deg_inv_list, w_up_list, w_down_list


    def forward(self, x, M, pe):
        """
        Args:
            x: [N, in_channels]
            M: (edge_indices_list, edge_weights_list)
            pe: [N, dirs]
        """
        N = x.size(0)
        x_trans = self.lin_in(x)

        # 预计算 deg_inv 和方向权重
        deg_inv_list, w_up_list, w_down_list = self.precompute(M, pe, N)

        edge_indices_list, edge_weights_list = M
        final_output = torch.zeros(N, self.out_channels, device=x.device)

        # Hop attention
        d_weight = torch.softmax(self.d, dim=0)

        for p in range(self.P):
            edge_index_p = edge_indices_list[p]
            topo_weight_p = edge_weights_list[p]
            deg_inv = deg_inv_list[p]
            w_up = w_up_list[p]
            w_down = w_down_list[p]

            row, col = edge_index_p
            E = row.size(0)

            # --- Channel 1: Self ---
            self_feat = x_trans

            # --- Channel 2: Avg ---
            msg_avg = x_trans[col] * topo_weight_p.unsqueeze(-1)
            out_avg = scatter_add(msg_avg, row, dim=0, dim_size=N)
            out_avg = out_avg * deg_inv.unsqueeze(-1)

            # --- Channel 3 & 4: Dir+/Dir− (一次 scatter_add) ---
            W = torch.cat([w_up, w_down], dim=1)  # [E, 2*dirs]

            # x_j * W → [E, out_channels, 2*dirs]
            msg_dir = x_trans[col].unsqueeze(-1) * W.unsqueeze(-2)

            # reshape → [E, out_channels * (2*dirs)]
            msg_dir = msg_dir.reshape(E, self.out_channels * (2 * self.dirs))

            # scatter_add → [N, out_channels * (2*dirs)]
            out_dir = scatter_add(msg_dir, row, dim=0, dim_size=N)
            out_dir = out_dir * deg_inv.unsqueeze(-1)

            # --- 拼接所有通道 ---
            h_concat = torch.cat([self_feat, out_avg, out_dir], dim=-1)

            # --- Hop fusion ---
            h_p = self.hop_fusions[p](h_concat)
            h_p = F.leaky_relu(h_p)

            # --- Hop attention ---
            final_output += h_p * d_weight[p] + self.hop_bias[p]

        return final_output


class DVDGN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=3,
        dropout=0.5,
        k=3,  # dfg k
        p=3,  # hop p
        dirs=3,
        add_self_loops: bool = True,
    ):
        super(DVDGN, self).__init__()
        self.in_channels = max(in_channels, 1)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k  # k farthest
        self.p = p  # p hop
        self.dirs = dirs
        self.add_self_loops = add_self_loops

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
        self.embedding = nn.Linear(
            in_features=self.in_channels, out_features=self.hidden_channels
        )

    def _build_convs(self):
        convs = nn.ModuleList()
        for _ in range(self.num_layers):
            convs.append(
                OPDMPConv(
                    self.hidden_channels, self.hidden_channels, P=self.p, dirs=self.dirs
                )
            )

        return convs

    def _build_feature_convs(self):
        convs = nn.ModuleList()
        for _ in range(self.num_layers):
            convs.append(
                OPDMPConv(
                    self.hidden_channels, self.hidden_channels, P=self.p, dirs=self.dirs
                )
            )
        return convs

    def _build_graph_norms(self):
        graph_norms = nn.ModuleList()
        for i in range(self.num_layers):
            graph_norms.append(GraphNorm(self.hidden_channels))
        return graph_norms

    def _build_auxiliary_graph(self, x, batch):
        return k_farthest_graph(
            x, self.k, batch, loop=True, cosine=True, direction=True
        )

    def process_phop(self, x, edge_index, source=None, mode="A"):
        if source is not None:
            if mode == "A":
                prefix = "a"
            else:
                prefix = "u"
            edge_index_l = []
            edge_weight_l = []
            for p in range(self.p):
                edge_index_l.append(source[f"{prefix}_{p}_edge_index"])
                edge_weight_l.append(source[f"{prefix}_{p}_edge_weight"])
            return edge_index_l, edge_weight_l

        N = x.size(0)
        # A = to_dense_adj(edge_index, max_num_nodes=N)[0]  # 稠密邻接矩阵 [N, N]

        if self.add_self_loops:
            # A = A + torch.eye(N, device=A.device)  # 自环
            edge_index, _ = add_self_loops(edge_index)

        # A_phop = []
        # for p in range(1, self.p + 1):
        #     Ap = torch.matrix_power(A, p)
        #     A_phop.append(dense_to_sparse(Ap))

        # return A_phop
        if mode == "A":
            return compute_A_phop(edge_index, N, self.p)
        elif mode == "U":
            return compute_U_phop(edge_index, N, self.p)

    # @torch.compile
    def forward(self, x, edge_index, batch, source=None, pe=None, *args, **kwargs):
        originl_x = x
        x = self.embedding(x)

        feature_graph_edge_index = self._build_auxiliary_graph(x, batch)
        SMp = self.process_phop(x, edge_index, source, mode="U")
        FMp = self.process_phop(x, feature_graph_edge_index, mode="U")

        all_x = []

        for i in range(self.num_layers):
            prev_x = x

            x = self.convs[i](x, M=SMp, pe=pe)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

            feature_x = self.feature_convs[i](x, M=FMp, pe=pe)
            feature_x = self.feature_norms[i](feature_x, batch)
            feature_x = F.leaky_relu(feature_x)
            # feature_x = F.dropout(feature_x, p=self.dropout, training=self.training)

            combined = torch.cat([x, feature_x], dim=-1)
            gate = torch.sigmoid(self.fusion_gate_linear(combined))

            fusion_x = gate * x + (1 - gate) * feature_x + prev_x
            fusion_x = F.dropout(fusion_x, p=self.dropout, training=self.training)

            # fusion_x = self.fusion_dense_mlp(combined) + prev_x
            all_x.append(fusion_x)
            x = fusion_x

        # graph_feature = 0
        # for i in range(self.num_layers):
        #     graph_feature += global_mean_pool(all_x[i - 1], batch)
        graph_feature = global_mean_pool(all_x[-1], batch)
        return graph_feature, 0
