"""
DP class-prototype exchange + cross-silo contrastive alignment.

The cross-silo alignment primitive of FedMAGNN:

  After each round, every client releases TWO 128-dim vectors — the
  DP-noised mean embedding of its benign nodes and of its malicious nodes.
  No raw features, no topology, no per-node embeddings ever leave a silo.

  In the next round, each client adds an InfoNCE alignment term that pulls
  its node embeddings toward OTHER silos' same-class prototypes and pushes
  them from opposite-class prototypes. This aligns the latent spaces across
  silos so the aggregated encoder learns silo-invariant notions of
  "malicious", enabling zero-shot recognition of attack families a silo has
  never observed locally.

Privacy: each prototype is one Gaussian-mechanism release of a clipped mean
(sensitivity clip_norm / n_class); accounted alongside the model-update DP.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .dp import dp_noised_mean_embedding


@torch.no_grad()
def compute_dp_prototypes(
    node_emb: torch.Tensor,
    labels: torch.Tensor,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[int, torch.Tensor]:
    """Return {class_id: DP-noised, L2-normalized prototype} for classes 0/1."""
    protos = {}
    for c in (0, 1):
        mask = labels == c
        if mask.sum() == 0:
            continue
        proto = dp_noised_mean_embedding(
            node_emb[mask].detach(),
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            generator=generator,
        )
        protos[c] = F.normalize(proto, dim=0).cpu()
    return protos


class PrototypeAlignmentLoss(torch.nn.Module):
    """
    InfoNCE between a client's node embeddings and other silos' prototypes.

    For node i with label y_i:
      positives = same-class prototypes from other silos
      negatives = opposite-class prototypes from other silos
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        node_emb: torch.Tensor,           # (N, D)
        labels: torch.Tensor,             # (N,)
        foreign_prototypes: List[Dict[int, torch.Tensor]],  # one dict per other silo
    ) -> torch.Tensor:
        if not foreign_prototypes:
            return node_emb.new_zeros(())

        proto_vecs, proto_labels = [], []
        for protos in foreign_prototypes:
            for c, vec in protos.items():
                proto_vecs.append(vec.to(node_emb.device))
                proto_labels.append(c)
        if len(set(proto_labels)) < 2:
            # Need both classes present among foreign prototypes for a
            # meaningful pull/push signal.
            return node_emb.new_zeros(())

        P = torch.stack(proto_vecs)                            # (M, D)
        pl = torch.tensor(proto_labels, device=node_emb.device)

        z = F.normalize(node_emb, dim=1)
        sim = z @ P.T / self.temperature                       # (N, M)
        pos_mask = (labels.unsqueeze(1) == pl.unsqueeze(0)).float()

        # Skip nodes with no positive prototype (class absent across silos)
        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return node_emb.new_zeros(())

        exp_sim = sim[has_pos].exp()
        pos_sum = (exp_sim * pos_mask[has_pos]).sum(dim=1)
        total = exp_sim.sum(dim=1)
        return -torch.log(pos_sum / (total + 1e-12) + 1e-12).mean()
