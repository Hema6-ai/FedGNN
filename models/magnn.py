"""
Base MAGNN Model — faithful reproduction of the paper architecture.

Architecture summary:
  1. Input projection: linear(in_dim → hidden_dim)
  2. Multi-scale GCN backbone: 3 × GCNConv layers with skip connections
     - Each layer aggregates 1-hop neighbors; 3 layers → 3-hop receptive field
     - The paper states "2-hop neighborhood" which likely refers to the
       Dirichlet-sampled local subgraph size, not the GCN depth. We implement
       3 GCNConv layers with skip connections as the most natural reading.
  3. Multi-head attention pooling: 4-head attention over node features
     (implemented as GATConv on the graph, giving per-node attention-weighted
      aggregations that drive the MHHC loss)
  4. Node-level outputs:
     - Classification head → logits (N, 2)
     - Regression head → malicious score (N,) in [0,1]
  5. Forward returns all intermediate representations needed by the loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class MAGNNBackbone(nn.Module):
    """3-layer GCN backbone with skip connections and BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_c, hidden_channels, add_self_loops=True))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.skips.append(
                nn.Linear(in_c, hidden_channels, bias=False)
                if in_c != hidden_channels
                else nn.Identity()
            )

        self.dropout = nn.Dropout(dropout)
        self.out_channels = hidden_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        layer_outs = []
        h = x
        for conv, norm, skip in zip(self.convs, self.norms, self.skips):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new + skip(h))    # skip connection
            h_new = self.dropout(h_new)
            layer_outs.append(h_new)
            h = h_new
        return h, layer_outs


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head graph attention on top of GCN backbone (H=4 heads).
    Produces per-head node embeddings for the MHHC loss.
    """

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = out_channels // num_heads
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=head_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.out_channels = head_dim * num_heads
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_concat: (N, H*head_dim) concatenated multi-head output
            h_per_head: (H, N, head_dim) per-head embeddings for MHHC loss
        """
        # GATConv with concat=True returns (N, H*head_dim) and stores attention weights
        h_concat = self.gat(x, edge_index)                         # (N, H*d)
        h_concat = self.norm(h_concat)

        # Reshape to (H, N, d) for per-head access
        N = h_concat.size(0)
        head_dim = h_concat.size(1) // self.num_heads
        h_per_head = h_concat.view(N, self.num_heads, head_dim).permute(1, 0, 2)  # (H, N, d)

        return h_concat, h_per_head


class MAGNN(nn.Module):
    """
    Full MAGNN model.

    Forward outputs a dict with all tensors needed by the loss functions.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.backbone = MAGNNBackbone(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mha = MultiHeadGraphAttention(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Classification head: node-level binary prediction
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Regression head: soft malicious probability score
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            x: node features (N, in_dim)
            edge_index: (2, E)
            edge_attr: edge features (E, edge_dim) — not used in base GCN
            batch: node-to-graph mapping (N,) for global pooling; None → single graph
        Returns:
            dict with keys:
              logits       (N, num_classes)
              scores       (N,)
              node_emb     (N, hidden_dim)  — final GCN+GAT embedding
              head_emb     (H, N, d)        — per-head embeddings for MHHC
              layer_embs   list of (N, hidden_dim) per GCN layer
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = F.relu(self.in_proj(x))                               # (N, hidden_dim)
        h_gcn, layer_embs = self.backbone(h, edge_index)          # (N, H_dim), [...]
        h_att, h_per_head = self.mha(h_gcn, edge_index)           # (N, H_dim), (H, N, d)

        logits = self.cls_head(h_att)                              # (N, 2)
        scores = self.reg_head(h_att).squeeze(-1)                  # (N,)

        return {
            "logits": logits,
            "scores": scores,
            "node_emb": h_att,
            "head_emb": h_per_head,
            "layer_embs": layer_embs,
            "gcn_emb": h_gcn,
        }
