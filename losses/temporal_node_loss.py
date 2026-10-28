"""
Temporal-Node Contrastive Loss (Eq. 3 in MAGNN paper).

For each node v, the model produces embeddings h_v^t1 and h_v^t2 from two
consecutive time-window graph snapshots. The loss encourages the same node's
embeddings to be close (positive pair) while pushing apart different nodes
(negative pairs) — standard InfoNCE / NT-Xent formulation.

L_TN = -1/N * Σ_v log [
    exp(sim(h_v^t1, h_v^t2) / τ) /
    Σ_{u≠v} exp(sim(h_v^t1, h_u^t2) / τ)
]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNodeLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, max_samples: int = 4096):
        super().__init__()
        self.temperature = temperature
        # The similarity matrix is N x N; at real traffic-graph sizes
        # (31k-134k nodes) that is 4-72 GB. Subsample anchors so InfoNCE
        # runs over at most max_samples matched pairs.
        self.max_samples = max_samples

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h1: node embeddings from snapshot t1, shape (N, D)
            h2: node embeddings from snapshot t2, shape (N, D)
                N nodes must correspond between h1 and h2 (same IP set).
        Returns:
            scalar loss
        """
        if h1.size(0) > self.max_samples:
            idx = torch.randperm(h1.size(0), device=h1.device)[: self.max_samples]
            h1, h2 = h1[idx], h2[idx]

        h1 = F.normalize(h1, dim=-1)  # (N, D)
        h2 = F.normalize(h2, dim=-1)  # (N, D)

        # Cosine similarity matrix: (N, N)
        sim = torch.mm(h1, h2.T) / self.temperature

        # Positive pairs are on the diagonal (same node across time)
        N = h1.size(0)
        labels = torch.arange(N, device=h1.device)

        # Cross-entropy with positives on diagonal
        loss = F.cross_entropy(sim, labels)
        return loss
