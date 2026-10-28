"""
Federated server: round loop, delta aggregation, DP, prototype exchange.

One round:
  1. broadcast global shared params to all clients
  2. each client trains `local_epochs` locally (with FedProx / alignment terms)
  3. each client's shared-param delta is (optionally) clipped
  4. server averages deltas (weighted by train-node count, or uniform under DP*)
  5. (optional) Gaussian noise added to the average  → new global shared params
  6. each client releases DP class prototypes → broadcast for next round

*Under DP the average is uniform: weighted averaging with data-dependent
 weights would leak silo sizes and complicate sensitivity analysis.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .client import FedClient
from .dp import GaussianDPMechanism


class FedServer:
    def __init__(
        self,
        clients: List[FedClient],
        rounds: int = 30,
        local_epochs: int = 2,
        dp: Optional[GaussianDPMechanism] = None,
        use_alignment: bool = False,
        zeroshot_silo: Optional[str] = None,
        logger=None,
    ):
        self.clients = clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.dp = dp
        self.use_alignment = use_alignment
        # Zero-shot experiment: prototypes are exchanged in BOTH arms (the
        # protocol is identical); the arms differ only in whether the
        # alignment loss is applied (clients' align_weight).
        self.exchange_prototypes = use_alignment or zeroshot_silo is not None
        self.zeroshot_silo = zeroshot_silo
        self.log = logger.info if logger else print

        # Initialize global shared params from client 0 and sync everyone,
        # so all silos start from an identical shared encoder.
        self.global_shared: Dict[str, torch.Tensor] = clients[0].get_shared_state()
        for c in clients:
            c.set_shared_state(self.global_shared)

        self.prototypes: Dict[str, dict] = {}
        self.history: List[dict] = []
        self.best_mean_f1 = -1.0
        self.best_round = -1
        self.best_client_states: Optional[List[dict]] = None

    # ── Aggregation ──────────────────────────────────────────────────────────

    def _aggregate(self, deltas: List[Dict[str, torch.Tensor]], weights: List[float]) -> None:
        if self.dp is not None and self.dp.enabled:
            deltas = [self.dp.clip_delta(d) for d in deltas]
            weights = [1.0] * len(deltas)  # uniform under DP (see module docstring)

        wsum = float(sum(weights))
        avg = {
            k: sum(w * d[k].float() for w, d in zip(weights, deltas)) / wsum
            for k in deltas[0]
        }
        if self.dp is not None:
            avg = self.dp.noise_aggregate(avg, num_clients=len(deltas))

        self.global_shared = {
            k: (self.global_shared[k].float() + avg[k]).to(self.global_shared[k].dtype)
            for k in self.global_shared
        }

    # ── Round loop ───────────────────────────────────────────────────────────

    def run(self) -> dict:
        for rnd in range(1, self.rounds + 1):
            t0 = time.time()
            deltas, weights, losses = [], [], {}

            for client in self.clients:
                client.set_shared_state(self.global_shared)
                foreign = None
                if self.exchange_prototypes:
                    foreign = [p for name, p in self.prototypes.items() if name != client.name]
                stats = client.local_train(
                    local_epochs=self.local_epochs,
                    global_shared=self.global_shared,
                    foreign_prototypes=foreign,
                )
                losses[client.name] = stats
                local = client.get_shared_state()
                deltas.append({
                    k: local[k].float() - self.global_shared[k].float()
                    for k in self.global_shared
                })
                weights.append(float(client.data.num_train_nodes))

            self._aggregate(deltas, weights)

            # Prototype exchange for the NEXT round (from post-aggregation encoders)
            if self.exchange_prototypes:
                for client in self.clients:
                    client.set_shared_state(self.global_shared)
                    self.prototypes[client.name] = client.release_prototypes()

            # Validation across all silos
            val = {}
            for client in self.clients:
                client.set_shared_state(self.global_shared)
                val[client.name] = client.evaluate_split("val")
            mean_f1 = float(np.mean([m["f1"] for m in val.values()]))
            mean_acc = float(np.mean([m["accuracy"] for m in val.values()]))

            eps = self.dp.epsilon() if (self.dp and self.dp.enabled) else None
            elapsed = time.time() - t0
            self.log(
                f"Round {rnd:03d}/{self.rounds} | mean val_f1={mean_f1:.2f}% "
                f"acc={mean_acc:.2f}% | "
                + " ".join(f"{n}={m['f1']:.1f}" for n, m in val.items())
                + (f" | eps={eps:.2f}" if eps is not None else "")
                + f" | {elapsed:.1f}s"
            )

            self.history.append({
                "round": rnd,
                "losses": losses,
                "val": val,
                "mean_val_f1": mean_f1,
                "mean_val_acc": mean_acc,
                "epsilon": eps,
            })

            if mean_f1 > self.best_mean_f1:
                self.best_mean_f1 = mean_f1
                self.best_round = rnd
                self.best_client_states = [
                    copy.deepcopy({k: v.cpu() for k, v in c.model.state_dict().items()})
                    for c in self.clients
                ]

        # ── Final test at best round ─────────────────────────────────────────
        if self.best_client_states is not None:
            for c, sd in zip(self.clients, self.best_client_states):
                c.model.load_state_dict(sd)

        test = {c.name: c.evaluate_split("test") for c in self.clients}
        mean_test_f1 = float(np.mean([m["f1"] for m in test.values()]))
        mean_test_acc = float(np.mean([m["accuracy"] for m in test.values()]))
        self.log(f"\nBest round: {self.best_round} (mean val F1 {self.best_mean_f1:.2f}%)")
        self.log(f"TEST | mean f1={mean_test_f1:.2f}% acc={mean_test_acc:.2f}%")
        for n, m in test.items():
            self.log(f"  {n:<10} acc={m['accuracy']:.2f} f1={m['f1']:.2f} "
                     f"prec={m['precision']:.2f} rec={m['recall']:.2f}")

        # Zero-shot evaluation: the attack-blind silo classifies its test set
        # purely from other silos' prototypes.
        zeroshot = None
        if self.zeroshot_silo is not None:
            zs_client = next(c for c in self.clients if c.name == self.zeroshot_silo)
            foreign = [p for name, p in self.prototypes.items() if name != self.zeroshot_silo]
            zeroshot = zs_client.evaluate_zeroshot(foreign, split="test")
            self.log(f"\nZERO-SHOT ({self.zeroshot_silo}, prototype-NN): {json.dumps(zeroshot)}")

        return {
            "test_metrics": test,
            "mean_test_f1": mean_test_f1,
            "mean_test_acc": mean_test_acc,
            "best_round": self.best_round,
            "best_mean_val_f1": self.best_mean_f1,
            "final_epsilon": self.dp.epsilon() if (self.dp and self.dp.enabled) else None,
            "zeroshot": zeroshot,
            "history": self.history,
        }
