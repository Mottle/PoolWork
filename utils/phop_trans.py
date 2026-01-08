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