import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import MixHopConv, APPNP, GCN2Conv, global_mean_pool, TransformerConv
import torch_geometric.nn
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F


class BaseLineRc(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        backbone="gcn",
        dropout=0.5,
        embed: bool = True,
        norm: bool = False,
        use_pe: bool = False,
    ):
        super(BaseLineRc, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.backbone = backbone
        self.dropout = dropout
        self.embed = embed
        self.norm = norm
        self.use_pe = use_pe

        if self.in_channels < 1:
            self.in_channels = 1

        if self.embed:
            self.embedding = nn.Linear(self.in_channels, hidden_channels)
            self.in_channels = hidden_channels

        self.build_convs()
        
        if self.norm:
            self.build_norms()

        self.build_pe()

    def build_convs(self):
        if self.num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(
                    self.build_conv(self.in_channels, self.hidden_channels)
                )
            else:
                self.convs.append(
                    self.build_conv(self.hidden_channels, self.hidden_channels)
                )

    def build_conv(self, in_channels, out_channels):
        if self.backbone == "mix_hop":
            p = 3
            powers = [*range(p)]
            return torch_geometric.nn.Sequential('x, edge_index', [
                (MixHopConv(in_channels, in_channels, powers=powers), 'x, edge_index -> x1'),
                (lambda x: x.view(x.size(0), 3, out_channels).mean(dim = 1))
            ])
        elif self.backbone == "appnp":
            return APPNP(K=3, alpha=0.5, dropout=self.dropout)
        elif self.backbone == 'gcn2':
            return GCN2Conv(out_channels, alpha=0.5)
        else:
            raise ValueError(f"backbone invalid: {self.backbone}")

    def build_norms(self):
        self.norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.norms.append(GraphNorm(self.hidden_channels))

    def build_pe(self, pe_raw_dim: int = 20):
        # self._lap_pe = LaplacianPE(pe_raw_dim)
        self._pe_linear = nn.Linear(pe_raw_dim, self.hidden_channels)
        self._pe_norm = torch.nn.LayerNorm(self.hidden_channels)
        self.pe_conv = nn.Sequential(
            # (self._lap_pe, 'x, edge_index, batch -> x'),
            self._pe_linear,
            self._pe_norm
        )

    def forward(self, x, edge_index, batch, pe = None, *args, **kwargs):
        originl_x = x
        if self.embed:
            x = self.embedding(x)
            ori_emb_x = x
        
        if pe is not None and self.use_pe:
            x = x + pe

        feature_all = []
        for i in range(self.num_layers):

            if self.backbone == 'graph_gps':
                x = self.convs[i](x, edge_index, batch)
            elif self.backbone == 'gcn2':
                x = self.convs[i](x, ori_emb_x, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            
            if self.norm:
                x = self.norms[i](x)

            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            feature = global_mean_pool(x, batch)
            feature_all.append(feature)
        # merge_feature = torch.mean(torch.stack(feature_all, dim=0), dim=0)
        merge_feature = feature_all[-1]

        return merge_feature, 0
