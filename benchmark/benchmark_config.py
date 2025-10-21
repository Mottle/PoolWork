import torch
import numpy as np
import random

class BenchmarkConfig:
    hidden_channels: int = 64
    num_layers: int = 3
    backbone: str = 'gcn'
    pooler: str = 'topk'
    graph_norm: bool = True
    batch_size: int = 128
    epochs: int = 100
    use_simple_datasets: bool = False
    catch_error: bool = False
    early_stop: bool = True
    seed: int = None
    kfold: int = 10

    def apply_random_seed(self):
        if self.seed:
            local_seed = self.seed
        else:
            local_seed = 0

        torch.manual_seed(local_seed)
        torch.cuda.manual_seed(local_seed)
        torch.cuda.manual_seed_all(local_seed)
        np.random.seed(local_seed)
        random.seed(local_seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    def format(self) -> str:
        return 'config setting:\n' + (f'hid_dim: {self.hidden_channels}\n' +
               f'layers: {self.num_layers}\n' +
               f'backbone: {self.backbone}\n' +
               f'pooler: {self.pooler}\n' +
               f'graph_norm: {self.graph_norm}\n' +
               f'batch_size: {self.batch_size}\n' +
               f'epochs: {self.epochs}\n' +
               f'kfold: {self.kfold}\n' +
               f'seed: {self.seed}\n' +
               f'catch_error: {self.catch_error}\n' +
               f'early_stop: {self.early_stop}\n' +
               f'use_simple_datasets: {self.use_simple_datasets}\n'
               )