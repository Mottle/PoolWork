from turtle import forward
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data

class PhopTrans(BaseTransform):    
    def __init__(self, p: int = 3):
        super().__init__()
        self.p = p
    
    def __call__(self, data: Data) -> Data:
        N = data.num_nodes
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            A = to_dense_adj(data.edge_index, max_num_nodes=N)[0]
            A_loop = A + torch.eye(N, device=A.device)
            Apop, Apop_w = [], []
            Apop_loop, Apop_loop_w = [], []
            for p in range(1, self.p + 1):
                Ap = torch.matrix_power(A, p)
                edge_index_p, edge_weight_p = dense_to_sparse(Ap)
                Apop.append(edge_index_p)
                Apop_w.append(edge_weight_p)

                Ap_loop = torch.matrix_power(A_loop, p)
                edge_index_p, edge_weight_p = dense_to_sparse(Ap_loop)
                Apop_loop.append(edge_index_p)
                Apop_loop_w.append(edge_weight_p)
            
            Apop, Apop_w = torch.tensor(Apop), torch.tensor(Apop_w)
            Apop_loop, Apop_loop_w = torch.tensor(Apop_loop), torch.tensor(Apop_loop_w)

            data.Aphop = torch.tensor([Apop, Apop_w])
            data.Aphop_loop = torch.tensor([Apop_loop, Apop_loop_w])
        else:
            data.Aphop = None
            data.Aphop_loop = None
        data.has_dense_adj = True
        return data
    
    # def __repr__(self) -> str:
    #     return f'{self.__class__.__name__}()'
    
    def forward(self, data):
        return self.__call__(data)
    
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg as sla
import torch


class LaplacianPETransform(BaseTransform):
    def __init__(self, k=20, normalization="sym"):
        self.k = k
        self.normalization = normalization

    def compute_lap_pe(self, edge_index, num_nodes):
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, normalization=self.normalization, num_nodes=num_nodes
        )

        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes=num_nodes)

        k = min(self.k, num_nodes - 2)
        if k <= 0:
            return torch.zeros((num_nodes, self.k))

        eigvals, eigvecs = sla.eigsh(L, k=k, which="SM")
        pe = torch.from_numpy(eigvecs).float()

        if pe.size(1) < self.k:
            pad = torch.zeros(num_nodes, self.k - pe.size(1))
            pe = torch.cat([pe, pad], dim=1)

        return pe

    def __call__(self, data):
        if hasattr(data, "pe"):
            return data

        pe = self.compute_lap_pe(data.edge_index, data.num_nodes)
        data.pe = pe
        return data

    def forward(self, data):
        return self.__call__(data)
