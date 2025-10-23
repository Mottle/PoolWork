import torch
import numpy as np
import random

class BenchmarkConfig:
    def __init__(self):
        self.hidden_channels: int = 64
        self.num_layers: int = 3
        self.backbone: str = 'gcn'
        self.pooler: str = 'topk'
        self.graph_norm: bool = True
        self.batch_size: int = 128
        self.epochs: int = 100
        self.use_simple_datasets: bool = False
        self.catch_error: bool = False
        self.early_stop: bool = True
        self.early_stop_epochs: int = 50
        self.seed: int = None
        self.kfold: int = 10
        self.lr = 0.001

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
               f'layers: {self.num_layers} ' +
               f'backbone: {self.backbone} ' +
               f'pooler: {self.pooler} ' +
               f'graph_norm: {self.graph_norm} ' +
               f'batch_size: {self.batch_size}\n' +
               f'epochs: {self.epochs} ' +
               f'kfold: {self.kfold} ' +
               f'seed: {self.seed} ' +
            #    f'catch_error: {self.catch_error} ' +
               f'early_stop: {self.early_stop} ' +
               f'use_simple_datasets: {self.use_simple_datasets}\n'
               )