import torch
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATConv,
    GATv2Conv,
    global_mean_pool,
    TransformerConv,
)
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F
from phop import PHopGCNConv, PHopGINConv, PHopLinkRWConv, PHopLinkGCNConv


class BaseLine(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        backbone="gcn",
        dropout=0.5,
        embed: bool = False,
        norm: bool = False,
    ):
        super(BaseLine, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.backbone = backbone
        self.dropout = dropout
        self.embed = embed
        self.norm = norm

        if self.in_channels < 1:
            self.in_channels = 1

        if self.embed:
            self.embedding = nn.Linear(self.in_channels, hidden_channels)
            self.in_channels = hidden_channels

        self.build_convs()
        if self.norm:
            self.build_norms()

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
        if self.backbone == "gcn":
            return GCNConv(in_channels, out_channels)
        elif self.backbone == "gin":
            # fnn = nn.Sequential(
            #     nn.Linear(in_channels, out_channels),
            #     nn.LeakyReLU(),
            #     nn.Linear(out_channels, out_channels),
            #     nn.LeakyReLU(),
            #     nn.Dropout(p=self.dropout)
            # )
            # fnn = nn.Linear(in_channels, out_channels)
            # fnn = nn.Sequential(
            #     nn.Linear(in_channels, out_channels),
            #     nn.LeakyReLU(),
            #     nn.Linear(out_channels, out_channels),
            #     nn.Dropout(p=self.dropout)
            # )
            fnn = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LeakyReLU(),
                nn.Linear(out_channels, out_channels),
            )
            return GINConv(fnn)
        elif self.backbone == "gat":
            heads = 4
            return GATConv(
                in_channels, out_channels // heads, heads=heads, dropout=self.dropout
            )
        elif self.backbone == "quad":
            from utils.quadratic.quadratic import QuadraticLayer

            # fnn = QuadraticLayer(in_channels, out_channels)
            # fnn = nn.Sequential(
            #     QuadraticLayer(in_channels, out_channels),
            #     nn.LeakyReLU(),
            #     QuadraticLayer(out_channels, out_channels),
            #     # nn.Dropout(p=self.dropout)
            # )
            # fnn = nn.Sequential(
            #     QuadraticLayer(in_channels, out_channels),
            #     nn.LeakyReLU(),
            #     nn.Linear(out_channels, out_channels),
            #     # nn.Dropout(p=self.dropout)
            # )
            # fnn = nn.Sequential(
            #     nn.Linear(in_channels, out_channels),
            #     nn.LeakyReLU(),
            #     QuadraticLayer(out_channels, out_channels),
            # )
            fnn = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.GELU(),
                QuadraticLayer(out_channels, out_channels),
            )
            return GINConv(fnn)
        elif self.backbone == "gt":
            heads = 4
            return TransformerConv(
                in_channels,
                out_channels,
                heads=4,
                dropout=self.dropout,
                beta=True,
                concat=False,
            )
        elif self.backbone == "phop_gcn":
            return PHopGCNConv(in_channels, out_channels, p=2)
        elif self.backbone == "phop_gin":
            return PHopGINConv(in_channels, out_channels, p=2)
        elif self.backbone == "phop_linkgcn":
            return PHopLinkGCNConv(in_channels, out_channels, P=2)
        elif self.backbone == 'gat_v2':
            heads = 4
            return GATv2Conv(
                in_channels, out_channels // heads, heads=heads, dropout=min(self.dropout * 2, 0.5)
            )
        else:
            raise ValueError(f"backbone invalid: {self.backbone}")

    def build_norms(self):
        self.norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.norms.append(GraphNorm(self.hidden_channels))

    def forward(self, x, edge_index, batch, *args, **kwargs):
        if self.embed:
            x = self.embedding(x)

        feature_all = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if self.norm:
                x = self.norms[i](x)
            x = F.leaky_relu(x)
            feature = global_mean_pool(x, batch)
            feature_all.append(feature)
        # merge_feature = torch.sum(torch.stack(feature_all, dim=0), dim=0)

        return feature_all[-1], 0
