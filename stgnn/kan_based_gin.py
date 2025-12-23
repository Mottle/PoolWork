import context
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from utils.kan.kan_based_gin import KANBasedGINConv

class KANBasedGIN(nn.Module):
    """
    Multi-layer KAN-GIN Network for node or graph classification.
    """
    def __init__(self, in_channels, hidden_channels, out_channels = None, num_layers = 3, **kan_kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        # 定义 KAN 参数
        kan_params = {
            'grid_size': 5,
            'spline_order': 3,
            'scale_noise': 0.1,
            'scale_base': 1.0,
            'scale_spline': 1.0,
            **kan_kwargs
        }

        if out_channels is None:
            out_channels = hidden_channels
        
        # 1. 第一层
        self.convs.append(
            KANBasedGINConv(in_channels, hidden_channels, hidden_channels, kan_params)
        )
        
        # 2. 中间层
        for _ in range(num_layers - 2):
            self.convs.append(
                KANBasedGINConv(hidden_channels, hidden_channels, hidden_channels, kan_params)
            )
        
        # 3. 最后一层 (通常最后一层不需要激活，但 GIN 通常都用，所以这里保持一致)
        self.convs.append(
            KANBasedGINConv(hidden_channels, out_channels, hidden_channels, kan_params)
        )
        
        # 可选：如果用于图分类，则需要一个 Readout 层和最终的分类器
        # 这里假设是节点分类，直接使用最后一层输出。

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # 只有中间层使用 LeakyReLU 或其它非线性，但 KAN 内部已含激活，
            # 这里为避免过度复杂化，直接依赖 KAN 内部的非线性。
            # 最后一个 KANGINConv 的输出作为最终结果。
            if i < self.num_layers - 1:
                 x = F.leaky_relu(x) # 仍然加入一个传统的ReLU或跳跃连接，以增加非线性组合
                #  x = F.dropout(x, p=0.5, training=self.training)
        
        global_feature = torch_geometric.nn.global_mean_pool(x, batch)  # 图级任务时使用

        return global_feature, 0