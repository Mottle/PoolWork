import torch
from torch import nn
from torch_geometric.nn import GCNConv

class MamboPooling(nn.Module):
    def __init__(self, in_dim, ratio = 0.5):
        super(MamboPooling, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio

        self.g1 = GCNConv(in_dim, in_dim)