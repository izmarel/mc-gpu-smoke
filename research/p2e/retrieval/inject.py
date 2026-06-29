"""Memory retrieval-augmentation glue (the "memory" encoder input).

Wraps the built FaissNetworkXStore graph. Each collection step, per env:
  1. RETRIEVE the K most-similar past states, follow each one's RESULT edge to
     "what came next", stack those next-state vectors most-similar-first into a
     fixed (K * latent_dim,) block (zero-padded if fewer than K, or empty store).
     -> this is what gets injected into DreamerV3's encoder as the "memory" key.
  2. STORE the current latent as a MemoryNode and add a RESULT edge from the
     previous step's node (building the state->next-state episodic graph).

Retrieve happens BEFORE store, so a step never retrieves itself. Per-env previous
node ids are tracked so RESULT edges follow real trajectories; a done resets them.

This is the literal "inject what happened last time" mechanism, on the real graph.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import Dict, List, Optional

import numpy as np

# NOTE: faiss runs multi-threaded here. The faiss+torch OpenMP double-load crash is
# fixed at the ENVIRONMENT level (conda torch + faiss-cpu share one OpenMP), not by
# pinning faiss to 1 thread. See TODO.md (Reference > Environment).

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.memory import (  # noqa: E402
    Edge,
    EdgeType,
    FaissNetworkXStore,
    MemoryNode,
)


class MemoryInjector:
    """Builds the fixed-size "memory" vector per step from the experience graph.

    Args:
        latent_dim: dimensionality of the latent vectors stored/retrieved.
        k: number of past neighbours to retrieve (memory block = k * latent_dim).
        n_envs: number of parallel environments (each has its own trajectory).
    """

    def __init__(self, latent_dim: int, k: int = 4, n_envs: int = 1,
                 max_nodes: Optional[int] = None) -> None:
        self.latent_dim = int(latent_dim)
        self.k = int(k)
        self.n_envs = int(n_envs)
        self.mem_dim = self.k * self.latent_dim
        self.store = FaissNetworkXStore()
        # bound the store to the most-recent N nodes (FIFO) so a long run can't OOM.
        # None / unset = unbounded; env MEM_INJECT_MAX overrides when not given.
        self.max_nodes = (
            int(max_nodes) if max_nodes is not None
            else (int(os.environ["MEM_INJECT_MAX"]) if os.environ.get("MEM_INJECT_MAX") else None)
        )
        self._order: "deque[str]" = deque()  # node ids in insertion order
        self._prev: List[Optional[str]] = [None] * self.n_envs
        self._t = 0

    def load_store(self, path: str) -> None:
        """Load a persisted store and resync FIFO order (for same-run resume / transfer)."""
        self.store.load(path)
        self._order = deque(self.store.memory_node_ids())
        self._prev = [None] * self.n_envs

    def _maybe_evict(self) -> None:
        """Keep only the most-recent max_nodes nodes (batched FIFO)."""
        if not self.max_nodes:
            return
        slack = max(1, self.max_nodes // 10)
        if len(self._order) <= self.max_nodes + slack:
            return
        n_evict = len(self._order) - self.max_nodes
        victims = set(self._order.popleft() for _ in range(n_evict))
        self.store.delete_nodes(victims, rebuild=True)
        for env in range(self.n_envs):
            if self._prev[env] in victims:
                self._prev[env] = None

    # --- retrieval ------------------------------------------------------------

    def _retrieve_one(self, latent: np.ndarray) -> np.ndarray:
        """Return a (k*latent_dim,) block: the next-states of the k nearest past states."""
        block = np.zeros((self.k, self.latent_dim), dtype=np.float32)
        hits = self.store.query_similar(latent, self.k)  # (node_id, distance), most-similar first
        for i, (node_id, _dist) in enumerate(hits[: self.k]):
            result_edges = self.store.get_edges(node_id, EdgeType.RESULT)
            if not result_edges:
                continue  # no recorded "next" -> leave zeros for this slot
            nxt = self.store.get_node(result_edges[0].target_id)
            if isinstance(nxt, MemoryNode) and nxt.dim == self.latent_dim:
                block[i] = nxt.state_vector
        return block.reshape(-1)

    # --- storage --------------------------------------------------------------

    def _store_one(self, latent: np.ndarray, env: int) -> None:
        node = MemoryNode(state_vector=latent, prediction_error=0.0, timestamp=float(self._t))
        nid = self.store.add_node(node)
        self._order.append(nid)
        if self._prev[env] is not None:
            self.store.add_edge(Edge(self._prev[env], nid, EdgeType.RESULT))
        self._prev[env] = nid

    # --- per-step API ---------------------------------------------------------

    def step(self, latents: np.ndarray, dones: Optional[np.ndarray] = None) -> np.ndarray:
        """latents: (n_envs, latent_dim) -> memory block (n_envs, k*latent_dim).

        Retrieves from the graph built so far, then stores the current latents.
        """
        latents = np.asarray(latents, dtype=np.float32).reshape(self.n_envs, self.latent_dim)
        out = np.zeros((self.n_envs, self.mem_dim), dtype=np.float32)
        for env in range(self.n_envs):
            out[env] = self._retrieve_one(latents[env])      # retrieve BEFORE storing self
            self._store_one(latents[env], env)
        self._t += 1
        if dones is not None:
            for env in range(self.n_envs):
                if dones[env]:
                    self._prev[env] = None                   # trajectory break
        self._maybe_evict()
        return out

    def stats(self) -> Dict:
        return {
            "nodes": self.store.node_count(),
            "t": self._t,
            "mem_dim": self.mem_dim,
        }
