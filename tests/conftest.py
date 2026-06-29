"""Shared fixtures and fake-data generation for the memory tests (TODO 1.8).

The generator builds a realistic-shaped memory graph entirely from synthetic
data so the storage/retrieval/signal layers can be exercised without any real
sensors or trained encoders:

* 500 ``MemoryNode`` state vectors, most drawn tightly around a handful of
  cluster centres (so similarity search has structure to find) plus a spray of
  uniform-random "background" vectors (so empty/novel regions exist);
* ``ACTION`` edges from each state to a synthetic ``ActionNode``;
* ``RESULT`` edges chaining states within a cluster into episodes.

Everything is seeded, so runs are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pytest

from src.memory import (
    ActionNode,
    Edge,
    EdgeType,
    InMemoryStore,
    MemoryNode,
    MemoryStore,
)

#: Dimensionality of the fake state vectors. Small for fast tests; the real
#: system uses 768/1792. The storage layer is dimension-agnostic.
DIM = 64
#: Total number of memory nodes generated (TODO 1.8 asks for 500).
N_NODES = 500
#: Number of intentional clusters in vector space.
N_CLUSTERS = 5
#: Std-dev of the gaussian blob around each cluster centre. Small so clusters
#: stay tight in the 64-dim space (perturbation norm ~ CLUSTER_SPREAD*sqrt(DIM)).
CLUSTER_SPREAD = 0.01
#: Fraction of nodes that are uniform-random background (rest are clustered).
BACKGROUND_FRACTION = 0.1
SEED = 1234


@dataclass
class FakeMemory:
    """A generated memory graph plus the ground-truth used to build it.

    Attributes:
        store: Populated :class:`InMemoryStore`.
        cluster_centers: ``(N_CLUSTERS, DIM)`` array of the centres used.
        cluster_node_ids: For each cluster, the ids of its member ``MemoryNode``s
            in chronological (RESULT-chain) order.
        dim: Vector dimensionality.
    """

    store: MemoryStore
    cluster_centers: np.ndarray
    cluster_node_ids: List[List[str]]
    dim: int

    def empty_region_vector(self) -> np.ndarray:
        """Return a unit-norm vector deliberately far from every cluster.

        Built by negating the mean cluster direction, so it points away from all
        of them — a reliably novel query for the "empty space" tests.
        """
        away = -self.cluster_centers.mean(axis=0)
        norm = np.linalg.norm(away)
        if norm == 0:
            away = np.ones(self.dim, dtype=np.float32)
            norm = np.linalg.norm(away)
        return (away / norm).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def generate_fake_memory(
    n_nodes: int = N_NODES,
    dim: int = DIM,
    n_clusters: int = N_CLUSTERS,
    seed: int = SEED,
    store_factory: Callable[[], MemoryStore] = InMemoryStore,
) -> FakeMemory:
    """Build and populate a :class:`MemoryStore` with synthetic experience.

    ``store_factory`` lets the same fake graph be built against any backend
    (default :class:`InMemoryStore`); the FAISS+NetworkX suite passes its own.
    """
    rng = np.random.default_rng(seed)
    store = store_factory()

    centers = _normalize_rows(rng.standard_normal((n_clusters, dim)))

    n_background = int(round(n_nodes * BACKGROUND_FRACTION))
    n_clustered = n_nodes - n_background

    cluster_node_ids: List[List[str]] = [[] for _ in range(n_clusters)]
    timestamp = 1_000_000.0  # arbitrary unix-ish base

    # Each cluster is a familiar region: a characteristic low prediction error
    # with little spread (consistent outcomes => high confidence). Background
    # nodes (below) stay high/random (surprising, unfamiliar).
    cluster_base_error = rng.uniform(0.05, 0.2, size=n_clusters)

    # --- clustered nodes: gaussian blobs around each centre -------------------
    for i in range(n_clustered):
        c = i % n_clusters
        vec = centers[c] + rng.standard_normal(dim) * CLUSTER_SPREAD
        vec = _unit(vec).astype(np.float32)
        error = float(np.clip(cluster_base_error[c] + rng.normal(0.0, 0.02), 0.0, 1.0))
        node = MemoryNode(
            state_vector=vec,
            prediction_error=error,
            timestamp=timestamp,
            text_metadata=f"cluster {c} sample {i}",
        )
        store.add_node(node)
        cluster_node_ids[c].append(node.node_id)
        _attach_action(store, node, rng)
        timestamp += 1.0

    # --- background nodes: uniform-random, far from clusters ------------------
    for _ in range(n_background):
        vec = _unit(rng.standard_normal(dim)).astype(np.float32)
        node = MemoryNode(
            state_vector=vec,
            prediction_error=float(rng.uniform(0.0, 1.0)),
            timestamp=timestamp,
            text_metadata="background",
        )
        store.add_node(node)
        _attach_action(store, node, rng)
        timestamp += 1.0

    # --- RESULT chains: wire each cluster's members into an episode -----------
    for ids in cluster_node_ids:
        for a, b in zip(ids, ids[1:]):
            store.add_edge(Edge(source_id=a, target_id=b, edge_type=EdgeType.RESULT))

    return FakeMemory(
        store=store,
        cluster_centers=centers.astype(np.float32),
        cluster_node_ids=cluster_node_ids,
        dim=dim,
    )


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _attach_action(store: InMemoryStore, node: MemoryNode, rng) -> None:
    """Create an ActionNode and an ACTION edge from ``node`` to it."""
    action = ActionNode(
        action_vector=rng.standard_normal(4).astype(np.float32),
        action_type=rng.choice(["up", "down", "left", "right"]),
    )
    store.add_node(action)
    store.add_edge(
        Edge(source_id=node.node_id, target_id=action.node_id, edge_type=EdgeType.ACTION)
    )


@pytest.fixture
def fake_memory() -> FakeMemory:
    """Pytest fixture: a freshly generated fake memory graph (TODO 1.8)."""
    return generate_fake_memory()
