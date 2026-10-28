"""
Graph Transformer layer for MAGNN-Plus.

Uses PyG's TransformerConv which implements:
  h_i' = W1*h_i + Σ_{j∈N(i)} α_{ij} * (W2*h_j + W3*e_{ij})
  α_{ij} = softmax( (W4*h_i)^T (W5*h_j + W3*e_{ij}) / √d )

This is strictly more expressive than standard GCNConv because:
  1. Attention weights are computed per-edge (adaptive aggregation)
  2. Edge features are directly incorporated into the aggregation
  3. Multi-head attention captures diverse structural patterns

Design note: The paper uses standard GCN for the base MAGNN and we replace it with
TransformerConv here in MAGNN-Plus. The interface is kept identical to GCNConv for
drop-in substitution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,      # if True, output = num_heads * (out_channels // num_heads)
        residual: bool = True,
    ):
        super().__init__()
        head_dim = out_channels // num_heads
        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=head_dim,
            heads=num_heads,
            concat=concat,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True,             # learnable residual gate inside attention
        )
        self.out_dim = head_dim * num_heads if concat else head_dim

        self.norm = nn.LayerNorm(self.out_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
        )
        self.norm2 = nn.LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(dropout)

        # Dimension adapter for residual connection
        self.residual = residual
        self.res_proj = (
            nn.Linear(in_channels, self.out_dim, bias=False)
            if in_channels != self.out_dim and residual
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = self.res_proj(x) if self.residual else None

        # Attention aggregation
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h + residual if residual is not None else h)

        # Feed-forward
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h


class MultiScaleGraphTransformer(nn.Module):
    """
    Stack of Graph Transformer layers with multi-scale feature collection.
    Concatenates outputs from all layers (similar to DenseNet), giving the
    downstream classifier access to multi-scale structural patterns.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        edge_dim: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        cur_in = in_channels
        for _ in range(num_layers):
            self.layers.append(
                GraphTransformerLayer(
                    in_channels=cur_in,
                    out_channels=hidden_channels,
                    edge_dim=edge_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    residual=True,
                )
            )
            cur_in = hidden_channels

        self.out_channels = hidden_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns:
            final node embeddings (N, hidden_channels)
            list of per-layer embeddings for multi-scale use
        """
        layer_outs = []
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            layer_outs.append(h)
        return h, layer_outs
