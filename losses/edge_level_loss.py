"""
Edge-Level Contrastive Loss (Eq. 4 in MAGNN paper).

Given an original graph view G and an augmented view G' (from EM or RWR),
the model produces edge embeddings e_{uv} and e'_{uv}. The loss encourages
matching edges in the two views to be similar (positive pairs) and non-matching
edges to be dissimilar (negative pairs).

Edge embeddings are computed as the concatenation (or element-wise product) of
the endpoint node embeddings, following common practice in graph contrastive learning.

L_EL = -1/E * Σ_{(u,v)} log [
    exp(sim(e_{uv}, e'_{uv}) / τ) /
    Σ_{(w,z)≠(u,v)} exp(sim(e_{uv}, e'_{wz}) / τ)
]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLevelLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, max_samples: int = 4096):
        super().__init__()
        self.temperature = temperature
        # The similarity matrix is E x E; real traffic graphs have 100k+
        # edges (26+ GB). Subsample matched edge pairs to at most max_samples.
        self.max_samples = max_samples

    @staticmethod
    def _edge_embeddings(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge embeddings as element-wise product of endpoint embeddings.
        Shape: (E, D)
        """
        src = edge_index[0]
        dst = edge_index[1]
        return h[src] * h[dst]

    def forward(
        self,
        h_orig: torch.Tensor,
        edge_index_orig: torch.Tensor,
        h_aug: torch.Tensor,
        edge_index_aug: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_orig: node embeddings from original graph (N, D)
            edge_index_orig: original edge connectivity (2, E)
            h_aug: node embeddings from augmented graph (N, D)
            edge_index_aug: augmented edge connectivity (2, E')
        Returns:
            scalar loss (computed on the min(E, E') matched edges)
        """
        e_orig = self._edge_embeddings(h_orig, edge_index_orig)   # (E, D)
        e_aug = self._edge_embeddings(h_aug, edge_index_aug)       # (E', D)

        # Align sizes by taking the minimum — edges beyond the smaller set are
        # dropped for this loss term (common approximation when graphs differ in size)
        min_e = min(e_orig.size(0), e_aug.size(0))
        if min_e == 0:
            return torch.tensor(0.0, device=h_orig.device, requires_grad=True)

        e_orig = e_orig[:min_e]
        e_aug = e_aug[:min_e]
        if min_e > self.max_samples:
            idx = torch.randperm(min_e, device=e_orig.device)[: self.max_samples]
            e_orig, e_aug = e_orig[idx], e_aug[idx]

        e_orig = F.normalize(e_orig, dim=-1)
        e_aug = F.normalize(e_aug, dim=-1)

        sim = torch.mm(e_orig, e_aug.T) / self.temperature
        labels = torch.arange(e_orig.size(0), device=e_orig.device)
        loss = F.cross_entropy(sim, labels)
        return loss
