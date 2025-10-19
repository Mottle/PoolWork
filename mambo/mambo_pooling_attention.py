from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import coalesce, scatter, softmax, remove_self_loops
from torch_geometric.nn import GATConv, GATv2Conv


class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor
    new_edge_score: Tensor


class MamboPoolingWithNodeAttention(torch.nn.Module):
    r"""The edge pooling operator with GAT-based edge scoring.

    This is a modified version of EdgePooling that uses GAT attention scores
    instead of a linear layer to compute edge scores.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (callable, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0.0`)
        add_to_edge_score (float, optional): A value to be added to each
            computed edge score. Adding this greatly helps with unpooling
            stability. (default: :obj:`0.5`)
        gat_heads (int, optional): Number of attention heads in GAT.
            (default: :obj:`1`)
        gat_concatenate (bool, optional): Whether to concatenate or average
            the attention heads. (default: :obj:`False`)
    """
    def __init__(
        self,
        in_channels: int,
        edge_score_method: Optional[Callable] = None,
        dropout: float = 0.2,
        add_to_edge_score: float = 0.0,
        gat_heads: int = 1,
        gat_concatenate: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout
        self.gat_heads = gat_heads
        self.gat_concatenate = gat_concatenate

        # Replace the linear layer with GAT convolution
        self.gat_conv = GATConv(
            in_channels,
            in_channels,  # Using same dimensions for output
            heads=gat_heads,
            concat=gat_concatenate,
            dropout=dropout,
            add_self_loops=True  # Keep self-loops for GAT
        )

        # If we're concatenating multiple heads, we need to reduce the dimension
        if gat_concatenate and gat_heads > 1:
            self.attention_reduce = torch.nn.Linear(in_channels * gat_heads, 1)
        else:
            self.attention_reduce = torch.nn.Linear(in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gat_conv.reset_parameters()
        self.attention_reduce.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Normalizes edge scores via softmax application."""
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via hyperbolic tangent application."""
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via sigmoid application."""
        return torch.sigmoid(raw_edge_score)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(torch.Tensor)* - The pooled node features.
            * **edge_index** *(torch.Tensor)* - The coarsened edge indices.
            * **batch** *(torch.Tensor)* - The coarsened batch vector.
            * **unpool_info** *(UnpoolInfo)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        edge_index, _ = remove_self_loops(edge_index)

        # Use GAT to compute node features and attention weights
        # We only need the attention weights for edge scoring
        gat_output, (edge_index_attn, attention_weights) = self.gat_conv(
            x, edge_index, return_attention_weights=True
        )
        
        # Extract edge scores from GAT attention
        # attention_weights shape: [num_edges, num_heads]
        if self.gat_heads > 1:
            if self.gat_concatenate:
                # If concatenating, we need to reduce multiple heads to a single score
                # First, we need to reshape and process through a linear layer
                e = self.attention_reduce(attention_weights).squeeze(-1)
            else:
                # If not concatenating, take the mean across heads
                e = attention_weights.mean(dim=1)
        else:
            # Single head, just squeeze
            e = attention_weights.squeeze(-1)

        # 移除自环
        _, e = remove_self_loops(edge_index_attn, e)
        
        # Apply dropout to edge scores
        e = F.dropout(e, p=self.dropout, training=self.training)
        
        # Apply the edge score normalization method
        e = self.compute_edge_score(e, edge_index, x.size(0))
        
        # Add constant to edge score for stability
        e = e + self.add_to_edge_score

        # Use the original EdgePooling merging logic
        x, edge_index, batch, unpool_info = self._merge_edges(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info

    def _merge_edges(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_score: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        """This method is kept unchanged from the original EdgePooling"""
        cluster = torch.empty_like(batch)
        perm: List[int] = torch.argsort(edge_score, descending=True).tolist()

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        mask = torch.ones(x.size(0), dtype=torch.bool)

        i = 0
        new_edge_indices: List[int] = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in perm:
            source = int(edge_index_cpu[0, edge_idx])
            if not bool(mask[source]):
                continue

            target = int(edge_index_cpu[1, edge_idx])
            if not bool(mask[target]):
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            mask[source] = False

            if source != target:
                cluster[target] = i
                mask[target] = False

            i += 1

        # The remaining nodes are simply kept:
        j = int(mask.sum())
        cluster[mask] = torch.arange(i, i + j, device=x.device)
        i += j

        # We compute the new features as an addition of the old ones.
        new_x = scatter(x, cluster, dim=0, dim_size=i, reduce='sum')
        new_edge_score = edge_score[new_edge_indices]
        if int(mask.sum()) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        new_edge_index = coalesce(cluster[edge_index], num_nodes=new_x.size(0))
        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster,
                                 batch=batch, new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(
        self,
        x: Tensor,
        unpool_info: UnpoolInfo,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (torch.Tensor): The node features.
            unpool_info (UnpoolInfo): Information that has been produced by
                :func:`EdgePooling.forward`.

        Return types:
            * **x** *(torch.Tensor)* - The unpooled node features.
            * **edge_index** *(torch.Tensor)* - The new edge indices.
            * **batch** *(torch.Tensor)* - The new batch vector.
        """
        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, heads={self.gat_heads})'