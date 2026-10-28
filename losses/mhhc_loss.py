"""
Multi-Head Hierarchical Contrastive (MHHC) Loss (Eq. 7 in MAGNN paper).

The MHHC loss operates at three granularities simultaneously:
  1. Node level  — contrastive between individual node embeddings per attention head
  2. Subgraph level — contrastive between local neighborhood aggregations per head
  3. Graph level  — contrastive between global graph embeddings per head

For H attention heads, each head produces its own view of the graph, and contrastive
learning is applied within and across heads. This forces different heads to capture
complementary structural patterns rather than collapsing to the same representation.

Eq. 7:
  L_MHHC = (1/H) * Σ_h [ λ_n * L_node^h + λ_s * L_sub^h + λ_g * L_graph^h ]

where each L_* is a standard InfoNCE loss between the two augmented views at that level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class MHHCLoss(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        temperature: float = 0.07,
        lambda_node: float = 1.0,
        lambda_sub: float = 0.5,
        lambda_graph: float = 0.5,
        max_samples: int = 4096,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.lambda_node = lambda_node
        self.lambda_sub = lambda_sub
        self.lambda_graph = lambda_graph
        # Node/subgraph InfoNCE builds N x N similarities; subsample nodes
        # (shared indices across heads) to keep memory bounded on large graphs.
        self.max_samples = max_samples

    @staticmethod
    def _infonce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """NT-Xent / InfoNCE between matched rows of z1 and z2."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        N = z1.size(0)
        if N == 0:
            return torch.tensor(0.0, device=z1.device)
        sim = torch.mm(z1, z2.T) / temperature
        labels = torch.arange(N, device=z1.device)
        return F.cross_entropy(sim, labels)

    def forward(
        self,
        head_outputs_v1: torch.Tensor,   # (H, N, D/H) per-head node embeddings, view 1
        head_outputs_v2: torch.Tensor,   # (H, N, D/H) per-head node embeddings, view 2
        batch: torch.Tensor,             # (N,) graph assignment for global pooling
        neighborhood_mask: torch.Tensor = None,  # (N, k) indices for subgraph aggregation
    ) -> torch.Tensor:
        """
        Args:
            head_outputs_v1: stacked per-head embeddings from augmented view 1, (H, N, d)
            head_outputs_v2: stacked per-head embeddings from augmented view 2, (H, N, d)
            batch: node-to-graph assignment tensor (N,)
            neighborhood_mask: optional (N, k) precomputed neighbor indices for subgraph pooling
        Returns:
            scalar MHHC loss
        """
        H = head_outputs_v1.size(0)
        N_all = head_outputs_v1.size(1)
        if N_all > self.max_samples:
            idx = torch.randperm(N_all, device=head_outputs_v1.device)[: self.max_samples]
            head_outputs_v1 = head_outputs_v1[:, idx, :]
            head_outputs_v2 = head_outputs_v2[:, idx, :]
            batch = batch[idx]
            if neighborhood_mask is not None:
                neighborhood_mask = neighborhood_mask[idx]
        total_loss = torch.tensor(0.0, device=head_outputs_v1.device)

        for h in range(H):
            z1 = head_outputs_v1[h]   # (N, d)
            z2 = head_outputs_v2[h]   # (N, d)

            # Node-level contrastive
            l_node = self._infonce(z1, z2, self.temperature)

            # Subgraph-level: mean of each node with its neighborhood proxy
            # When neighborhood_mask is absent, use the node embeddings directly
            # as a proxy for local neighborhood (simplified but functional)
            if neighborhood_mask is not None:
                # neighborhood_mask: (N, k) — indices of k neighbors per node
                N, k = neighborhood_mask.shape
                nb_idx = neighborhood_mask.clamp(0, z1.size(0) - 1)   # (N, k)
                sub1 = z1[nb_idx].mean(dim=1)   # (N, d)
                sub2 = z2[nb_idx].mean(dim=1)
            else:
                sub1 = z1
                sub2 = z2

            l_sub = self._infonce(sub1, sub2, self.temperature)

            # Graph-level: global mean pooling per graph
            g1 = global_mean_pool(z1, batch)   # (B, d)
            g2 = global_mean_pool(z2, batch)   # (B, d)
            l_graph = self._infonce(g1, g2, self.temperature)

            total_loss = total_loss + (
                self.lambda_node * l_node
                + self.lambda_sub * l_sub
                + self.lambda_graph * l_graph
            )

        return total_loss / H
