import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

class MergePooling(nn.Module):
    def __init__(self, channels: int):
        super(MergePooling, self).__init__()
        self.channels = channels
        self.channel_attention_linear = nn.Linear(in_features=channels * 2, out_features=channels)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean_pooled = global_mean_pool(x, batch)
        max_pooled = global_max_pool(x, batch)
        merged = torch.cat([mean_pooled, max_pooled], dim=1)
        alpha = torch.sigmoid(self.channel_attention_linear(merged))
        attended = alpha * mean_pooled + (1 - alpha) * max_pooled
        return attended