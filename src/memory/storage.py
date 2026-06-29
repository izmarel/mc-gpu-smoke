"""Storage layer for the unified vector-graph memory (POC 1, TODO 1.3).

This module defines the *backend-agnostic* storage interface
(:class:`MemoryStore`) and a dependency-free reference implementation
(:class:`InMemoryStore`) used for tests and as a behavioural contract.

The interface deliberately knows nothing about LadybugDB, FAISS, NetworkX or any
other engine. A concrete backend (e.g. ``LadybugStore``) only has to subclass
:class:`MemoryStore` and implement the abstract methods; everything downstream
(retrieval, confidence, novelty, learning progress) is written against the
abstract type and works unchanged.

Responsibilities of a store:

* hold nodes (:class:`~src.memory.models.MemoryNode` / ``ActionNode``) keyed by
  ``node_id`` and the typed edges between them;
* answer vector-similarity queries over ``MemoryNode`` state vectors;
* traverse typed edges (one hop and multi-hop chains);
* persist to disk and reload, surviving process restarts.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import (
    Edge,
    EdgeType,
    MemoryNode,
    Node,
    node_from_dict,
)

#: ``(node_id, distance)`` pair returned by similarity search. Smaller distance
#: means more similar.
SimilarityHit = Tuple[str, float]


class DistanceMetric(str, Enum):
    """Distance metric used by :meth:`MemoryStore.query_similar`."""

    #: 1 - cosine_similarity. Range [0, 2]; 0 == identical direction.
    COSINE = "cosine"
    #: Squared Euclidean (L2) distance. Range [0, inf).
    L2 = "l2"


class MemoryStore(ABC):
    """Abstract storage backend for the memory graph.

    A backend stores two things: nodes (keyed by ``node_id``) and typed,
    directed edges. ``MemoryNode`` state vectors additionally feed a vector
    index so similar past states can be retrieved quickly.

    Concrete implementations: :class:`InMemoryStore` (reference, dict-based) and
    a future ``LadybugStore`` (graph DB with native vector index).
    """

    # --- CRUD -----------------------------------------------------------------

    @abstractmethod
    def add_node(self, node: Node) -> str:
        """Insert ``node`` and (for ``MemoryNode``) index its state vector.

        Returns the stored node's ``node_id``. Re-adding an existing id
        overwrites the previous node.
        """

    @abstractmethod
    def add_edge(self, edge: Edge) -> None:
        """Insert a typed directed edge. Both endpoints must already exist."""

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        """Return the node with ``node_id``, or ``None`` if absent."""

    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Delete a node, its vector-index entry and all incident edges.

        Returns ``True`` if a node was removed, ``False`` if it was absent.
        """

    # --- vector search --------------------------------------------------------

    @abstractmethod
    def query_similar(
        self, state_vector: np.ndarray, k: int
    ) -> List[SimilarityHit]:
        """Return the ``k`` ``MemoryNode``s closest to ``state_vector``.

        Result is a list of ``(node_id, distance)`` sorted by ascending
        distance (most similar first). Only ``MemoryNode``s participate; action
        nodes are never returned. Returns fewer than ``k`` hits if the store
        holds fewer memory nodes.
        """

    # --- graph traversal ------------------------------------------------------

    @abstractmethod
    def get_edges(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        """Return outgoing edges from ``node_id``.

        If ``edge_type`` is given, only edges of that type are returned.
        """

    @abstractmethod
    def get_edge_chain(self, node_id: str, hops: int) -> List[MemoryNode]:
        """Follow RESULT edges from ``node_id`` for up to ``hops`` steps.

        Returns the chain of ``MemoryNode``s reached, *excluding* the start
        node: ``[next_state, next_next_state, ...]`` with at most ``hops``
        entries. Stops early when no further RESULT edge exists. This is the
        "episode of experience" walk: state -> result -> state -> result ...
        """

    # --- persistence ----------------------------------------------------------

    @abstractmethod
    def persist(self, path: str) -> None:
        """Write the full store (nodes + edges) to ``path`` so it survives a
        restart."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Replace the store's contents with data previously written by
        :meth:`persist`."""

    # --- convenience (non-abstract) ------------------------------------------

    def __len__(self) -> int:
        """Number of nodes currently stored. Override for efficiency."""
        return self.node_count()

    @abstractmethod
    def node_count(self) -> int:
        """Total number of nodes (memory + action) in the store."""


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``1 - cosine_similarity(a, b)`` in ``[0, 2]``.

    Zero vectors are treated as maximally dissimilar (distance 1.0) rather than
    raising, so degenerate inputs don't crash retrieval.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos = float(np.dot(a, b) / (na * nb))
    # Guard against tiny floating-point excursions outside [-1, 1].
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared Euclidean distance between ``a`` and ``b``."""
    diff = a - b
    return float(np.dot(diff, diff))


_METRIC_FNS = {
    DistanceMetric.COSINE: cosine_distance,
    DistanceMetric.L2: l2_distance,
}


class InMemoryStore(MemoryStore):
    """Dependency-free reference backend backed by plain dicts.

    Holds everything in RAM, runs similarity search as a brute-force scan, and
    persists to a single JSON file. It is the behavioural contract every real
    backend must satisfy and is fast enough for the POC-scale tests (hundreds to
    a few thousand nodes). It is *not* meant for production scale — that's what
    LadybugDB is for.

    Args:
        metric: Distance metric for :meth:`query_similar`. Cosine by default
            (direction matters more than magnitude for these state vectors).
    """

    def __init__(self, metric: DistanceMetric = DistanceMetric.COSINE) -> None:
        self.metric = DistanceMetric(metric)
        self._nodes: Dict[str, Node] = {}
        # Outgoing adjacency: source_id -> list of edges.
        self._edges: Dict[str, List[Edge]] = {}

    # --- CRUD -----------------------------------------------------------------

    def add_node(self, node: Node) -> str:
        self._nodes[node.node_id] = node
        return node.node_id

    def add_edge(self, edge: Edge) -> None:
        if edge.source_id not in self._nodes:
            raise KeyError(f"edge source {edge.source_id!r} not in store")
        if edge.target_id not in self._nodes:
            raise KeyError(f"edge target {edge.target_id!r} not in store")
        self._edges.setdefault(edge.source_id, []).append(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def delete_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        # Drop outgoing edges.
        self._edges.pop(node_id, None)
        # Drop incoming edges.
        for src, edges in self._edges.items():
            kept = [e for e in edges if e.target_id != node_id]
            if len(kept) != len(edges):
                self._edges[src] = kept
        return True

    # --- vector search --------------------------------------------------------

    def query_similar(
        self, state_vector: np.ndarray, k: int
    ) -> List[SimilarityHit]:
        if k <= 0:
            return []
        query = np.asarray(state_vector, dtype=np.float32)
        dist_fn = _METRIC_FNS[self.metric]
        hits: List[SimilarityHit] = []
        for node in self._nodes.values():
            if not isinstance(node, MemoryNode):
                continue
            if node.state_vector.shape != query.shape:
                continue
            hits.append((node.node_id, dist_fn(query, node.state_vector)))
        hits.sort(key=lambda h: h[1])
        return hits[:k]

    # --- graph traversal ------------------------------------------------------

    def get_edges(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        edges = self._edges.get(node_id, [])
        if edge_type is None:
            return list(edges)
        want = EdgeType(edge_type)
        return [e for e in edges if e.edge_type == want]

    def get_edge_chain(self, node_id: str, hops: int) -> List[MemoryNode]:
        chain: List[MemoryNode] = []
        current = node_id
        seen = {node_id}
        for _ in range(max(0, hops)):
            result_edges = self.get_edges(current, EdgeType.RESULT)
            if not result_edges:
                break
            nxt = result_edges[0].target_id
            if nxt in seen:  # guard against cycles
                break
            node = self._nodes.get(nxt)
            if not isinstance(node, MemoryNode):
                break
            chain.append(node)
            seen.add(nxt)
            current = nxt
        return chain

    # --- persistence ----------------------------------------------------------

    def persist(self, path: str) -> None:
        payload = {
            "metric": self.metric.value,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for src in self._edges.values() for e in src],
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        # Write atomically: temp file then rename, so a crash mid-write can't
        # corrupt an existing snapshot.
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.metric = DistanceMetric(payload.get("metric", DistanceMetric.COSINE))
        self._nodes = {}
        self._edges = {}
        for node_data in payload["nodes"]:
            node = node_from_dict(node_data)
            self._nodes[node.node_id] = node
        for edge_data in payload["edges"]:
            edge = Edge.from_dict(edge_data)
            self._edges.setdefault(edge.source_id, []).append(edge)

    # --- convenience ----------------------------------------------------------

    def node_count(self) -> int:
        return len(self._nodes)
