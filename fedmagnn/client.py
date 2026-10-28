"""
Federated client: one dataset silo running local MAGNN-Plus training.

Personalization split (FedPer-style):
  SHARED (aggregated across silos): in_proj, pe, gnn, head_projs — the graph
    encoder. These learn silo-invariant structure.
  PRIVATE (never leaves the silo): cls_head, reg_head, temporal_enc — the
    task heads adapt to each silo's local label balance and attack mix.

`share_mode="full"` shares everything instead (pure FedAvg baseline).

The local objective per epoch is the original MAGNN-Plus objective
(focal cls + MSE reg + integrated contrastive) plus two federated terms:
  + mu/2 · ||w_shared - w_global||²          (FedProx, if mu > 0)
  + lambda_align · L_proto_align             (cross-silo alignment, if enabled)
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from train import (
    build_model, build_scheduler, graph_to_device,
    apply_dirichlet_sampling, evaluate,
)
from utils.augmentation import edge_modification, random_walk_restart
from losses import IntegratedContrastiveLoss, MultiTaskLoss

from .partition import ClientData
from .prototypes import compute_dp_prototypes, PrototypeAlignmentLoss

ENCODER_PREFIXES = ("in_proj.", "pe.", "gnn.", "head_projs.")


class FedClient:
    def __init__(
        self,
        client_data: ClientData,
        cfg: dict,
        device: torch.device,
        share_mode: str = "encoder",      # "encoder" | "full"
        fedprox_mu: float = 0.0,
        align_weight: float = 0.0,
        align_temperature: float = 0.1,
        proto_clip: float = 1.0,
        proto_noise_multiplier: float = 0.0,
        seed: int = 42,
    ):
        self.data = client_data
        self.name = client_data.name
        self.cfg = cfg
        self.device = device
        self.share_mode = share_mode
        self.fedprox_mu = fedprox_mu
        self.align_weight = align_weight
        self.proto_clip = proto_clip
        self.proto_noise_multiplier = proto_noise_multiplier

        self.model = build_model("magnn_plus", cfg, client_data.in_dim).to(device)
        tr = cfg["training"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=tr.get("lr", 0.001),
            weight_decay=tr.get("weight_decay", 1e-5),
        )

        lc = cfg.get("loss", {})
        self.contrastive_loss_fn = IntegratedContrastiveLoss(
            temperature=lc.get("temperature", 0.07),
            lambda1=lc.get("lambda1", 1.0),
            lambda2=lc.get("lambda2", 1.0),
            lambda3=lc.get("lambda3", 1.0),
            num_heads=cfg["model"].get("num_heads", 4),
        ).to(device)
        self.task_loss_fn = MultiTaskLoss(
            alpha=lc.get("alpha", 0.5), beta=lc.get("beta", 0.5), label_smoothing=0.0,
        )
        self.align_loss_fn = PrototypeAlignmentLoss(temperature=align_temperature)

        self._proto_gen = torch.Generator().manual_seed(seed + hash(self.name) % 10000)
        self._epoch_counter = 0

        # Graphs stay on CPU; moved to device per use (keeps GPU memory bounded
        # with 4 clients sharing one device).
        self.train_graph_cpu = client_data.train_graph
        self.val_graph_cpu = client_data.val_graph
        self.test_graph_cpu = client_data.test_graph
        self.snapshots_cpu = client_data.train_snapshots

    # ── Shared-parameter plumbing ────────────────────────────────────────────

    def _is_shared(self, param_name: str) -> bool:
        if self.share_mode == "full":
            return True
        return param_name.startswith(ENCODER_PREFIXES)

    def get_shared_state(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
            if self._is_shared(k)
        }

    def set_shared_state(self, shared: Dict[str, torch.Tensor]) -> None:
        sd = self.model.state_dict()
        for k, v in shared.items():
            if k in sd:
                sd[k] = v.to(sd[k].device, dtype=sd[k].dtype)
        self.model.load_state_dict(sd)

    # ── Local training ───────────────────────────────────────────────────────

    def local_train(
        self,
        local_epochs: int,
        global_shared: Optional[Dict[str, torch.Tensor]] = None,
        foreign_prototypes: Optional[List[dict]] = None,
    ) -> dict:
        """Run `local_epochs` epochs; return loss stats from the last epoch."""
        tr = self.cfg["training"]
        train_graph = graph_to_device(self.train_graph_cpu, self.device)

        # FedProx anchor (shared params at round start), kept on device
        prox_anchor = None
        if self.fedprox_mu > 0 and global_shared is not None:
            prox_anchor = {
                k: v.to(self.device) for k, v in global_shared.items()
            }

        stats = {}
        for _ in range(local_epochs):
            self._epoch_counter += 1
            epoch = self._epoch_counter
            self.model.train()
            self.optimizer.zero_grad()

            aug_args = dict(
                drop_ratio=tr.get("augment_drop_ratio", 0.1),
                add_ratio=tr.get("augment_add_ratio", 0.1),
                seed=epoch,
            )
            rwr_args = dict(
                restart_prob=tr.get("rwr_restart_prob", 0.15),
                num_steps=tr.get("rwr_steps", 100),
                seed=epoch + 1000,
            )
            data_cpu = self.train_graph_cpu
            aug1 = graph_to_device(edge_modification(data_cpu, **aug_args), self.device)
            aug2 = graph_to_device(random_walk_restart(data_cpu, **rwr_args), self.device)

            sampled = apply_dirichlet_sampling(
                train_graph,
                size=tr.get("dirichlet_size", 30),
                alpha=tr.get("dirichlet_alpha", 1.0),
                seed=epoch,
            )

            # Snapshot embeddings for the temporal encoder
            snap_embs = None
            if len(self.snapshots_cpu) >= 2:
                self.model.eval()
                with torch.no_grad():
                    snap_embs = []
                    for snap in self.snapshots_cpu:
                        s = graph_to_device(snap, self.device)
                        out = self.model(s.x, s.edge_index, s.edge_attr)
                        snap_embs.append(out["gcn_emb"])
                    # Snapshots cover different node subsets; the temporal
                    # encoder stacks them, so truncate to the common size.
                    n_min = min(e.size(0) for e in snap_embs)
                    snap_embs = [e[:n_min] for e in snap_embs]
                self.model.train()

            out = self.model(sampled.x, sampled.edge_index, sampled.edge_attr, snapshots=snap_embs)
            out_aug1 = self.model(aug1.x, aug1.edge_index, aug1.edge_attr)
            out_aug2 = self.model(aug2.x, aug2.edge_index, aug2.edge_attr)

            N = min(out["node_emb"].size(0), out_aug1["node_emb"].size(0), out_aug2["node_emb"].size(0))
            batch = torch.zeros(N, dtype=torch.long, device=self.device)

            # Embeddings are truncated to N rows; drop edges referencing nodes >= N
            def _mask_edges(ei):
                keep = (ei[0] < N) & (ei[1] < N)
                return ei[:, keep]

            l_contrast = self.contrastive_loss_fn(
                h_t1=out["node_emb"][:N],
                h_t2=out_aug1["node_emb"][:N],
                h_orig=out["node_emb"][:N],
                edge_index_orig=_mask_edges(sampled.edge_index),
                h_aug=out_aug1["node_emb"][:N],
                edge_index_aug=_mask_edges(aug1.edge_index),
                head_v1=out["head_emb"][:, :N, :],
                head_v2=out_aug2["head_emb"][:, :N, :],
                batch=batch,
            )

            labels = sampled.y[:N]
            logits = out["logits"][:N]
            scores = out["scores"][:N]

            import os
            if os.environ.get("FEDMAGNN_PLAIN_CE"):   # ablation switch
                l_cls = torch.nn.functional.cross_entropy(logits, labels)
            else:
                l_cls = self.model.classification_loss(logits, labels)
            l_reg = torch.nn.functional.mse_loss(scores, labels.float())
            l_total = l_cls + self.task_loss_fn.alpha * l_reg + self.task_loss_fn.beta * l_contrast

            # Cross-silo prototype alignment
            l_align = logits.new_zeros(())
            if self.align_weight > 0 and foreign_prototypes:
                l_align = self.align_loss_fn(out["node_emb"][:N], labels, foreign_prototypes)
                l_total = l_total + self.align_weight * l_align

            # FedProx proximal term on shared params
            l_prox = logits.new_zeros(())
            if prox_anchor is not None:
                for k, p in self.model.named_parameters():
                    if k in prox_anchor:
                        l_prox = l_prox + (p - prox_anchor[k]).pow(2).sum()
                l_total = l_total + 0.5 * self.fedprox_mu * l_prox

            l_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            stats = {
                "total": float(l_total.detach()),
                "cls": float(l_cls.detach()),
                "reg": float(l_reg.detach()),
                "contrastive": float(l_contrast.detach()),
                "align": float(l_align.detach()),
                "prox": float(l_prox.detach()),
            }
        return stats

    # ── Prototype release ────────────────────────────────────────────────────

    @torch.no_grad()
    def release_prototypes(self) -> dict:
        """DP-noised class prototypes from the current local encoder."""
        self.model.eval()
        g = graph_to_device(self.train_graph_cpu, self.device)
        out = self.model(g.x, g.edge_index, g.edge_attr)
        return compute_dp_prototypes(
            out["node_emb"], g.y,
            clip_norm=self.proto_clip,
            noise_multiplier=self.proto_noise_multiplier,
            generator=self._proto_gen,
        )

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate_split(self, split: str = "val") -> dict:
        g = {"val": self.val_graph_cpu, "test": self.test_graph_cpu, "train": self.train_graph_cpu}[split]
        return evaluate(self.model, g, self.device)

    @torch.no_grad()
    def evaluate_zeroshot(self, foreign_prototypes: List[dict], split: str = "test") -> dict:
        """
        Prototype-nearest-neighbor classification: label each node by cosine
        similarity to the mean foreign benign / malicious prototype. This is
        how a silo that has NEVER seen an attack can detect one — entirely
        from other silos' DP prototype releases.
        """
        import torch.nn.functional as F
        from utils.metrics import compute_metrics

        protos = {0: [], 1: []}
        for p in foreign_prototypes:
            for c, vec in p.items():
                protos[c].append(vec)
        if not protos[0] or not protos[1]:
            return {"error": "need both classes among foreign prototypes"}
        centers = torch.stack([
            F.normalize(torch.stack(protos[0]).mean(dim=0), dim=0),
            F.normalize(torch.stack(protos[1]).mean(dim=0), dim=0),
        ]).to(self.device)                                      # (2, D)

        self.model.eval()

        def embed(graph):
            gd = graph_to_device(graph, self.device)
            out = self.model(gd.x, gd.edge_index, gd.edge_attr)
            return F.normalize(out["node_emb"], dim=1), gd.y

        z_test, y_test = embed({"val": self.val_graph_cpu, "test": self.test_graph_cpu}[split])
        sim = z_test @ centers.T                                # (N, 2)

        # Naive prototype-NN (argmax over foreign class centers)
        naive_preds = sim.argmax(dim=1)
        naive_score = torch.softmax(sim, dim=1)[:, 1]
        naive = compute_metrics(
            y_true=y_test.cpu().numpy(),
            y_pred_cls=naive_preds.cpu().numpy(),
            y_pred_score=naive_score.cpu().numpy(),
        )

        # Calibrated anomaly score: margin toward the foreign malicious
        # prototype, thresholded at the 95th percentile of the silo's OWN
        # (all-benign) training scores. Requires no attack knowledge.
        z_train, _ = embed(self.train_graph_cpu)
        train_margin = z_train @ centers[1] - z_train @ centers[0]
        threshold = torch.quantile(train_margin.float(), 0.95)
        test_margin = z_test @ centers[1] - z_test @ centers[0]
        cal_preds = (test_margin > threshold).long()
        cal_score = torch.sigmoid((test_margin - threshold) * 10.0)
        calibrated = compute_metrics(
            y_true=y_test.cpu().numpy(),
            y_pred_cls=cal_preds.cpu().numpy(),
            y_pred_score=cal_score.cpu().numpy(),
        )

        return {"naive": naive, "calibrated": calibrated,
                "threshold": float(threshold)}
