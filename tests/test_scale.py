"""Scale test: HNSW vs brute-force at 10k/100k/1M + NetworkX traversal/pickle (TODO 1.12).

Measures:
  - FAISS IndexHNSWFlat: insert time, query time, recall@10 vs brute-force
  - FAISS IndexFlatIP (brute-force): same — baseline for recall comparison
  - NetworkX: traversal speed, RAM usage, pickle save/load time
  - Identifies the REAL bottleneck

Run with:  pytest tests/test_scale.py -v -s
Skip slow:  pytest tests/test_scale.py -v -s -k "not 1m"
"""

from __future__ import annotations

import gc
import os
import time
import pickle
from typing import List

import faiss
import networkx as nx
import numpy as np
import pytest

from src.memory import FaissNetworkXStore, MemoryNode, Edge, EdgeType

DIM = 64  # realistic CLIP embedding dimension (smaller for speed)
K = 10    # recall@K


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gen_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Generate n unit-normalized random vectors (cosine similarity domain)."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    faiss.normalize_L2(vecs)
    return vecs


def _build_hnsw(vecs: np.ndarray, m=32, ef_construction=40, ef_search=128) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vecs.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vecs)
    return index


def _build_flat(vecs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def _recall_at_k(hnsw_hits: np.ndarray, flat_hits: np.ndarray, k: int = K) -> float:
    """Fraction of brute-force top-k that HNSW also returned."""
    recall = 0.0
    for i in range(len(hnsw_hits)):
        truth = set(flat_hits[i][:k].tolist())
        got = set(hnsw_hits[i][:k].tolist())
        recall += len(truth & got) / k
    return recall / len(hnsw_hits)


def _build_graph(n: int, vecs: np.ndarray) -> FaissNetworkXStore:
    """Build a FaissNetworkXStore with n MemoryNodes and a chain of RESULT edges."""
    store = FaissNetworkXStore()
    ids: List[str] = []
    for i in range(n):
        node = MemoryNode(state_vector=vecs[i], prediction_error=0.1, timestamp=float(i))
        store.add_node(node)
        ids.append(node.node_id)
    # Chain: node[i] -> node[i+1] via RESULT edge (every 10th node gets edges to keep it sparse)
    for i in range(0, n - 1, 10):
        store.add_edge(Edge(source_id=ids[i], target_id=ids[i+1], edge_type=EdgeType.RESULT))
    return store


# ---------------------------------------------------------------------------
# FAISS: HNSW vs brute-force
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("n", [10_000, 100_000], ids=["10k", "100k"])
def test_faiss_hnsw_vs_flat_recall_and_timing(n):
    """HNSW insert/query time and recall@10 vs brute-force at 10k and 100k."""
    vecs = _gen_vectors(n, DIM, seed=42)
    queries = _gen_vectors(100, DIM, seed=99)

    # --- HNSW ---
    t0 = time.perf_counter()
    hnsw = _build_hnsw(vecs)
    t_insert_hnsw = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, hnsw_hits = hnsw.search(queries, K)
    t_query_hnsw = time.perf_counter() - t0

    # --- Flat (brute-force) ---
    t0 = time.perf_counter()
    flat = _build_flat(vecs)
    t_insert_flat = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, flat_hits = flat.search(queries, K)
    t_query_flat = time.perf_counter() - t0

    recall = _recall_at_k(hnsw_hits, flat_hits, K)

    print(f"\n=== FAISS {n} nodes, dim={DIM} ===")
    print(f"  HNSW  insert: {t_insert_hnsw:.3f}s  query(100): {t_query_hnsw*1000:.1f}ms")
    print(f"  Flat  insert: {t_insert_flat:.3f}s  query(100): {t_query_flat*1000:.1f}ms")
    print(f"  Recall@{K}: {recall:.4f}")
    print(f"  HNSW speedup: {t_query_flat/t_query_hnsw:.1f}x")

    # HNSW must be fast
    assert t_query_hnsw < t_query_flat or n < 100_000, "HNSW should be faster at 100k+"
    # Recall must be decent (>0.8 at these scales with good params)
    assert recall > 0.80, f"Recall@{K} too low: {recall:.4f}"


@pytest.mark.slow
@pytest.mark.parametrize("n", [10_000, 100_000], ids=["10k", "100k"])
def test_faiss_insert_throughput(n):
    """Insert throughput: nodes/second the FaissNetworkXStore can ingest."""
    vecs = _gen_vectors(n, DIM, seed=7)
    store = FaissNetworkXStore()

    t0 = time.perf_counter()
    for i in range(n):
        store.add_node(MemoryNode(state_vector=vecs[i], prediction_error=0.1, timestamp=float(i)))
    elapsed = time.perf_counter() - t0

    throughput = n / elapsed
    print(f"\n=== INSERT THROUGHPUT {n} nodes ===")
    print(f"  Total: {elapsed:.2f}s  ({throughput:.0f} nodes/sec)")

    # At minimum should handle 30fps = 30 nodes/sec with headroom
    assert throughput > 30, f"Insert throughput too low: {throughput:.0f}/sec (need >30 for 30fps)"


# ---------------------------------------------------------------------------
# NetworkX: traversal + pickle
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("n", [10_000, 100_000], ids=["10k", "100k"])
def test_networkx_traversal_and_pickle(n):
    """NetworkX edge-chain traversal speed and pickle save/load time."""
    vecs = _gen_vectors(n, DIM, seed=11)
    store = _build_graph(n, vecs)

    # --- Traversal: get_edge_chain over 5 hops from first node ---
    first_id = store.memory_node_ids()[0]
    t0 = time.perf_counter()
    chain = store.get_edge_chain(first_id, hops=5)
    t_traverse = time.perf_counter() - t0

    print(f"\n=== NETWORKX {n} nodes ===")
    print(f"  5-hop traversal: {t_traverse*1000:.2f}ms  (chain len={len(chain)})")

    # --- Pickle save ---
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        pkl_path = tf.name

    t0 = time.perf_counter()
    store.persist(pkl_path)
    t_persist = time.perf_counter() - t0

    pkl_size = os.path.getsize(pkl_path) / (1024 * 1024)  # MB

    # --- Pickle load + index rebuild ---
    t0 = time.perf_counter()
    loaded = FaissNetworkXStore()
    loaded.load(pkl_path)
    t_load = time.perf_counter() - t0

    print(f"  Pickle save: {t_persist:.2f}s  ({pkl_size:.1f} MB)")
    print(f"  Pickle load: {t_load:.2f}s  (incl. FAISS index rebuild)")

    # --- Verify integrity ---
    assert loaded.node_count() == store.node_count(), "Node count mismatch after load"
    assert len(loaded.memory_node_ids()) == len(store.memory_node_ids())

    # Query still works after reload
    survivor = loaded.get_node(first_id)
    hits = loaded.query_similar(survivor.state_vector, k=1)
    assert hits, "Query returned nothing after reload"

    os.unlink(pkl_path)

    # Traversal must be sub-second
    assert t_traverse < 1.0, f"Traversal too slow: {t_traverse:.3f}s"


# ---------------------------------------------------------------------------
# 1M nodes — the real test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_scale_1m_faiss():
    """FAISS HNSW vs brute-force at 1M nodes — the real operational ceiling test."""
    n = 1_000_000
    vecs = _gen_vectors(n, DIM, seed=42)
    queries = _gen_vectors(100, DIM, seed=99)

    # --- HNSW ---
    t0 = time.perf_counter()
    hnsw = _build_hnsw(vecs, m=48, ef_construction=80, ef_search=512)
    t_insert_hnsw = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, hnsw_hits = hnsw.search(queries, K)
    t_query_hnsw = time.perf_counter() - t0

    # --- Flat (brute-force) ---
    t0 = time.perf_counter()
    flat = _build_flat(vecs)
    t_insert_flat = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, flat_hits = flat.search(queries, K)
    t_query_flat = time.perf_counter() - t0

    recall = _recall_at_k(hnsw_hits, flat_hits, K)

    print(f"\n=== FAISS 1M nodes, dim={DIM} ===")
    print(f"  HNSW  insert: {t_insert_hnsw:.1f}s  query(100): {t_query_hnsw*1000:.1f}ms")
    print(f"  Flat  insert: {t_insert_flat:.1f}s  query(100): {t_query_flat*1000:.1f}ms")
    print(f"  Recall@{K}: {recall:.4f}")
    print(f"  HNSW speedup: {t_query_flat/t_query_hnsw:.1f}x")

    # HNSW must be dramatically faster
    assert t_query_hnsw < t_query_flat, "HNSW must be faster at 1M"
    # Recall must still be usable
    assert recall > 0.60, f"Recall@{K} too low at 1M: {recall:.4f}"
    # Query must be sub-second for real-time use
    assert t_query_hnsw < 1.0, f"HNSW query too slow at 1M: {t_query_hnsw:.3f}s"

    del vecs, flat
    gc.collect()


@pytest.mark.slow
def test_scale_1m_networkx():
    """NetworkX graph + pickle save/load at 1M nodes — find the real bottleneck.

    Tests NetworkX directly (no FAISS) because the TODO asks for NetworkX
    traversal/pickle speed, not FAISS insert throughput.
    """
    n = 1_000_000

    # Build a raw NetworkX DiGraph with 1M nodes + chain edges (every 10th).
    # Store MemoryNode objects as attributes to mirror real usage.
    from src.memory import MemoryNode
    rng = np.random.default_rng(11)
    vecs = rng.standard_normal((n, DIM)).astype(np.float32)

    g = nx.MultiDiGraph()
    node_ids = []
    t0 = time.perf_counter()
    for i in range(n):
        nid = f"node-{i}"
        g.add_node(nid, obj=MemoryNode(state_vector=vecs[i], prediction_error=0.1, timestamp=float(i)))
        node_ids.append(nid)
    for i in range(0, n - 1, 10):
        g.add_edge(node_ids[i], node_ids[i+1], edge=Edge(source_id=node_ids[i], target_id=node_ids[i+1], edge_type=EdgeType.RESULT))
    t_build = time.perf_counter() - t0

    assert g.number_of_nodes() == n
    print(f"\n=== NETWORKX 1M nodes (raw, no FAISS) ===")
    print(f"  Build: {t_build:.1f}s")

    # --- Traversal: 5-hop chain ---
    t0 = time.perf_counter()
    chain = []
    current = node_ids[0]
    seen = {current}
    for _ in range(5):
        result_edges = [d["edge"] for _, _, d in g.out_edges(current, data=True) if d["edge"].edge_type == EdgeType.RESULT]
        if not result_edges:
            break
        nxt = result_edges[0].target_id
        if nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    t_traverse = time.perf_counter() - t0
    print(f"  5-hop traversal: {t_traverse*1000:.2f}ms  (chain len={len(chain)})")

    # --- Pickle save ---
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        pkl_path = tf.name
    t0 = time.perf_counter()
    with open(pkl_path, "wb") as fh:
        pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)
    t_persist = time.perf_counter() - t0
    pkl_size = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"  Pickle save: {t_persist:.1f}s  ({pkl_size:.0f} MB)")
    print(f"  Pickle size/node: {pkl_size*1024/n:.2f} KB/node")

    # --- Pickle load ---
    t0 = time.perf_counter()
    with open(pkl_path, "rb") as fh:
        g2 = pickle.load(fh)
    t_load = time.perf_counter() - t0
    print(f"  Pickle load: {t_load:.1f}s")

    assert g2.number_of_nodes() == n
    os.unlink(pkl_path)

    assert t_traverse < 1.0, f"Traversal too slow at 1M: {t_traverse:.3f}s"

    del vecs, g, g2
    gc.collect()