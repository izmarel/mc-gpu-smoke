"""FAISS + NetworkX storage backend for the memory graph (POC 1, TODO 1.3).

This is the production-shaped backend chosen in TODO 1.1 after LadybugDB was
benchmarked and rejected (too slow for vector search past ~10k nodes). It splits
the two jobs a :class:`~src.memory.storage.MemoryStore` has to do across two
specialised libraries and keeps them in sync on one shared key:

* **FAISS** — ``IndexHNSWFlat`` (approximate nearest neighbour) wrapped in an
  ``IndexIDMap2`` so vectors carry stable integer ids. Answers the
  vector-similarity query (:meth:`query_similar`). HNSW is sub-millisecond at
  100k+ where the brute-force reference backend is not.
* **NetworkX** — a ``MultiDiGraph`` holding every node (as the ``obj`` attribute)
  and every typed, directed edge. Answers CRUD and edge traversal.

The shared key is ``node_id``: each ``MemoryNode`` is a graph node *and* a row in
the FAISS index, mapped ``node_id <-> int64`` via :attr:`_id_to_int`.

Two FAISS realities shape the design:

* HNSW has **no in-place delete** — ``remove_ids`` raises. Deletion is therefore
  a *tombstone*: the int id is dropped from the live maps (so it can never be
  returned) and remembered in :attr:`_deleted_ints` only so queries know how far
  to over-fetch. When tombstones pile up the index is rebuilt from the live
  graph (:meth:`rebuild_index`); the prune cycle does this explicitly.
* HNSW is **approximate**, governed by ``M`` / ``efConstruction`` / ``efSearch``.
  The defaults (32 / 40 / 64) give effectively perfect recall at POC scale while
  staying fast; the scale test (TODO 1.12) measures recall against brute force.

The FAISS index is treated as a rebuildable *cache* of the graph: persistence
pickles only the graph + id maps (atomically), and :meth:`load` rebuilds the
index from the restored vectors. One file, one atomic rename, no two-file skew.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Optional, Set

import faiss
import networkx as nx
import numpy as np

from .models import ActionNode, Edge, EdgeType, MemoryNode, Node
from .storage import DistanceMetric, MemoryStore, SimilarityHit

#: Rebuild the FAISS index once tombstones reach this absolute count *and* exceed
#: this fraction of the live index. Keeps over-fetch bounded without rebuilding on
#: every single delete (which would make bulk pruning O(n^2)).
_REBUILD_MIN_TOMBSTONES = 256
_REBUILD_TOMBSTONE_FRACTION = 0.5


class FaissNetworkXStore(MemoryStore):
    """FAISS (vectors) + NetworkX (graph) backend (TODO 1.1 decision).

    Drop-in replacement for :class:`~src.memory.storage.InMemoryStore`: same
    abstract contract, same distance convention (cosine distance ``1 - cos`` in
    ``[0, 2]`` by default), but backed by an HNSW vector index and a real graph
    so it scales past the brute-force reference backend.

    Args:
        metric: Distance metric for :meth:`query_similar`. Cosine by default
            (direction matters more than magnitude for state vectors); cosine is
            implemented as inner product over L2-normalized vectors.
        m: HNSW graph degree (``M``). Higher = better recall, more memory.
        ef_construction: HNSW build-time search width. Higher = better graph
            quality, slower inserts.
        ef_search: HNSW query-time search width. Higher = better recall, slower
            queries. Raised automatically when a query asks for more neighbours.

    Notes:
        All ``MemoryNode`` state vectors in a single store must share one
        dimensionality — the index is fixed-dim, set lazily from the first memory
        node. A node of a different dimension is stored in the graph but not
        indexed (so it is simply never returned by similarity search, matching
        the reference backend's behaviour of skipping shape mismatches).
    """

    def __init__(
        self,
        metric: DistanceMetric = DistanceMetric.COSINE,
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 64,
    ) -> None:
        self.metric = DistanceMetric(metric)
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)

        # Graph holds the authoritative node objects and all typed edges.
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

        # FAISS state. The index is lazily created once we know the dimension.
        # ``_hnsw`` is the wrapped IndexHNSWFlat (kept so efSearch is tunable;
        # IndexIDMap2 doesn't expose ``.hnsw`` directly).
        self._index: Optional[faiss.Index] = None
        self._hnsw: Optional[faiss.IndexHNSWFlat] = None
        self._dim: Optional[int] = None
        self._id_to_int: Dict[str, int] = {}
        self._int_to_id: Dict[int, str] = {}
        self._next_int: int = 0
        #: Int ids still living in the FAISS index but already deleted from the
        #: graph. Only used to size query over-fetch and trigger rebuilds.
        self._deleted_ints: Set[int] = set()

    # --- index helpers --------------------------------------------------------

    def _faiss_metric(self) -> int:
        """Map our :class:`DistanceMetric` to a FAISS metric constant."""
        if self.metric is DistanceMetric.COSINE:
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2

    def _new_index(self, dim: int) -> faiss.Index:
        """Build an empty ``IndexIDMap2(IndexHNSWFlat)`` for ``dim`` dimensions.

        Stashes the wrapped HNSW index in :attr:`_hnsw` so ``efSearch`` stays
        tunable (``IndexIDMap2`` doesn't surface ``.hnsw``).
        """
        base = faiss.IndexHNSWFlat(dim, self.m, self._faiss_metric())
        base.hnsw.efConstruction = self.ef_construction
        base.hnsw.efSearch = self.ef_search
        self._hnsw = base
        return faiss.IndexIDMap2(base)

    def _prepare(self, vector: np.ndarray) -> np.ndarray:
        """Return a contiguous float32 copy, L2-normalized for cosine.

        Zero vectors are left as zeros (inner product 0 => cosine distance 1.0),
        matching :func:`~src.memory.storage.cosine_distance`'s zero-vector rule.
        """
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        if self.metric is DistanceMetric.COSINE:
            vec = vec.copy()
            faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    def _to_distance(self, score: float) -> float:
        """Convert a FAISS score into our distance convention.

        Cosine: FAISS returns inner product (== cosine sim for normalized
        vectors); distance is ``1 - sim``, clamped at 0 for tiny FP excursions.
        L2: FAISS already returns squared Euclidean distance.
        """
        if self.metric is DistanceMetric.COSINE:
            return max(0.0, 1.0 - float(score))
        return float(score)

    def _index_memory_node(self, node: MemoryNode) -> None:
        """Add (or re-add) ``node``'s vector to the FAISS index."""
        # Re-adding an existing id: tombstone the stale vector first.
        old_int = self._id_to_int.pop(node.node_id, None)
        if old_int is not None:
            self._int_to_id.pop(old_int, None)
            self._deleted_ints.add(old_int)

        if self._index is None:
            self._dim = node.dim
            self._index = self._new_index(self._dim)
        if node.dim != self._dim:
            # Mismatched dimension: keep it in the graph but out of the index, so
            # similarity search simply never returns it (reference-backend parity).
            return

        int_id = self._next_int
        self._next_int += 1
        self._id_to_int[node.node_id] = int_id
        self._int_to_id[int_id] = node.node_id
        vec = self._prepare(node.state_vector).reshape(1, -1)
        self._index.add_with_ids(vec, np.array([int_id], dtype=np.int64))

    def _maybe_rebuild(self) -> None:
        """Rebuild the index if tombstones have grown too large."""
        if self._index is None:
            return
        n_dead = len(self._deleted_ints)
        if n_dead < _REBUILD_MIN_TOMBSTONES:
            return
        if n_dead > _REBUILD_TOMBSTONE_FRACTION * max(1, self._index.ntotal):
            self.rebuild_index()

    def rebuild_index(self) -> None:
        """Rebuild the FAISS index from the live graph, dropping tombstones.

        Called automatically when deletions accumulate and explicitly by the
        prune cycle (TODO 1.11) after a bulk delete. Reuses each node's existing
        int id (ids stay stable across rebuilds) and preserves insertion order,
        so query results are unchanged.
        """
        if self._dim is None:
            return
        new_index = self._new_index(self._dim)
        if self._id_to_int:
            vectors = np.empty((len(self._id_to_int), self._dim), dtype=np.float32)
            ids = np.empty(len(self._id_to_int), dtype=np.int64)
            for row, (node_id, int_id) in enumerate(self._id_to_int.items()):
                node = self._graph.nodes[node_id]["obj"]
                vectors[row] = self._prepare(node.state_vector)
                ids[row] = int_id
            new_index.add_with_ids(vectors, ids)
        self._index = new_index
        self._deleted_ints.clear()

    # --- CRUD -----------------------------------------------------------------

    def add_node(self, node: Node) -> str:
        # Graph node carries the object; re-adding overwrites the attribute.
        self._graph.add_node(node.node_id, obj=node)
        if isinstance(node, MemoryNode):
            self._index_memory_node(node)
        return node.node_id

    def add_edge(self, edge: Edge) -> None:
        if edge.source_id not in self._graph:
            raise KeyError(f"edge source {edge.source_id!r} not in store")
        if edge.target_id not in self._graph:
            raise KeyError(f"edge target {edge.target_id!r} not in store")
        self._graph.add_edge(edge.source_id, edge.target_id, edge=edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        if node_id not in self._graph:
            return None
        return self._graph.nodes[node_id]["obj"]

    def delete_node(self, node_id: str) -> bool:
        if node_id not in self._graph:
            return False
        int_id = self._id_to_int.pop(node_id, None)
        if int_id is not None:
            self._int_to_id.pop(int_id, None)
            self._deleted_ints.add(int_id)
        # MultiDiGraph.remove_node drops the node and every incident edge.
        self._graph.remove_node(node_id)
        self._maybe_rebuild()
        return True

    def delete_nodes(self, node_ids: Iterable[str], rebuild: bool = True) -> int:
        """Delete many nodes, rebuilding the index at most once.

        The bulk path for the prune cycle (TODO 1.11): per-node
        :meth:`delete_node` could trigger several O(n) index rebuilds mid-loop;
        this tombstones everything then rebuilds once (when ``rebuild`` is True).
        Pass ``rebuild=False`` to defer the rebuild to the caller when staging
        several delete passes. Returns the number of nodes actually removed.
        """
        removed = 0
        for node_id in node_ids:
            if node_id not in self._graph:
                continue
            int_id = self._id_to_int.pop(node_id, None)
            if int_id is not None:
                self._int_to_id.pop(int_id, None)
                self._deleted_ints.add(int_id)
            self._graph.remove_node(node_id)
            removed += 1
        if rebuild and removed:
            self.rebuild_index()
        return removed

    # --- vector search --------------------------------------------------------

    def query_similar(
        self, state_vector: np.ndarray, k: int
    ) -> List[SimilarityHit]:
        if k <= 0:
            return []
        if self._index is None or self._index.ntotal == 0:
            return []
        query = np.asarray(state_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._dim:
            # Dimension mismatch: nothing comparable (reference-backend parity).
            return []

        # Over-fetch to cover tombstones still sitting in the index, then filter.
        oversample = min(self._index.ntotal, k + len(self._deleted_ints))
        # HNSW needs efSearch >= the number of results it must return.
        self._hnsw.hnsw.efSearch = max(self.ef_search, oversample)

        prepared = self._prepare(query).reshape(1, -1)
        scores, ids = self._index.search(prepared, oversample)

        hits: List[SimilarityHit] = []
        for int_id, score in zip(ids[0], scores[0]):
            if int_id == -1:
                continue
            node_id = self._int_to_id.get(int(int_id))
            if node_id is None:  # tombstoned vector, skip
                continue
            hits.append((node_id, self._to_distance(score)))
            if len(hits) >= k:
                break
        return hits

    # --- graph traversal ------------------------------------------------------

    def get_edges(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        if node_id not in self._graph:
            return []
        edges = [data["edge"] for _, _, data in self._graph.out_edges(node_id, data=True)]
        if edge_type is None:
            return edges
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
            node = self.get_node(nxt)
            if not isinstance(node, MemoryNode):
                break
            chain.append(node)
            seen.add(nxt)
            current = nxt
        return chain

    # --- persistence ----------------------------------------------------------

    def persist(self, path: str) -> None:
        """Atomically pickle the graph + id maps to ``path``.

        The FAISS index is a rebuildable cache and is *not* written — it is
        reconstructed from the restored vectors in :meth:`load`. Written to a
        temp file then ``os.replace``d so a crash mid-write can't corrupt an
        existing snapshot.
        """
        payload = {
            "metric": self.metric.value,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "dim": self._dim,
            "next_int": self._next_int,
            "id_to_int": self._id_to_int,
            "graph": self._graph,
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        """Replace contents from a :meth:`persist` snapshot and rebuild the index."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.metric = DistanceMetric(payload["metric"])
        self.m = payload["m"]
        self.ef_construction = payload["ef_construction"]
        self.ef_search = payload["ef_search"]
        self._dim = payload["dim"]
        self._next_int = payload["next_int"]
        self._id_to_int = payload["id_to_int"]
        self._graph = payload["graph"]
        self._int_to_id = {i: nid for nid, i in self._id_to_int.items()}
        self._deleted_ints = set()
        self._index = None
        # Rebuild the vector index from the restored node vectors.
        if self._dim is not None and self._id_to_int:
            self._index = self._new_index(self._dim)
            vectors = np.empty((len(self._id_to_int), self._dim), dtype=np.float32)
            ids = np.empty(len(self._id_to_int), dtype=np.int64)
            for row, (node_id, int_id) in enumerate(self._id_to_int.items()):
                node = self._graph.nodes[node_id]["obj"]
                vectors[row] = self._prepare(node.state_vector)
                ids[row] = int_id
            self._index.add_with_ids(vectors, ids)

    # --- convenience ----------------------------------------------------------

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def degree(self, node_id: str) -> int:
        """Total degree (incoming + outgoing edges) of ``node_id``, 0 if absent.

        Used by the prune cycle to spot highly connected "junction" nodes worth
        keeping (TODO 1.11).
        """
        if node_id not in self._graph:
            return 0
        return int(self._graph.degree(node_id))

    def memory_node_ids(self) -> List[str]:
        """Ids of every ``MemoryNode`` currently stored."""
        return [
            nid
            for nid, data in self._graph.nodes(data=True)
            if isinstance(data["obj"], MemoryNode)
        ]

    def action_node_ids(self) -> List[str]:
        """Ids of every ``ActionNode`` currently stored."""
        return [
            nid
            for nid, data in self._graph.nodes(data=True)
            if isinstance(data["obj"], ActionNode)
        ]
