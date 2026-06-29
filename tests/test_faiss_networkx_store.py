"""Contract tests for :class:`FaissNetworkXStore` (TODO 1.3).

The FAISS+NetworkX backend must satisfy the exact same behavioural contract as
the brute-force reference :class:`InMemoryStore`. These tests mirror the
storage/retrieval/signal coverage in ``test_memory.py`` but exercise the real
HNSW index + NetworkX graph, plus a few HNSW-specific concerns: tombstone
deletion, index rebuild, and pickle round-trips of the graph.

Run with: ``.venv/bin/pytest tests/test_faiss_networkx_store.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest

from src.memory import (
    ActionNode,
    Edge,
    EdgeType,
    FaissNetworkXStore,
    LinearLearningProgress,
    MemoryNode,
    RetrievalResult,
    Retriever,
    SimpleConfidence,
    SimpleNovelty,
)

from conftest import DIM, N_NODES, FakeMemory, generate_fake_memory


@pytest.fixture
def faiss_memory() -> FakeMemory:
    """A fake memory graph built against the FAISS+NetworkX backend."""
    return generate_fake_memory(store_factory=FaissNetworkXStore)


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


def test_add_and_get_node():
    store = FaissNetworkXStore()
    node = MemoryNode(state_vector=[1.0, 0.0], prediction_error=0.1, timestamp=1.0)
    store.add_node(node)
    fetched = store.get_node(node.node_id)
    assert fetched is node
    assert store.node_count() == 1
    assert len(store) == 1


def test_get_missing_node_returns_none():
    assert FaissNetworkXStore().get_node("nope") is None


def test_add_edge_requires_existing_endpoints():
    store = FaissNetworkXStore()
    a = MemoryNode(state_vector=[1.0], prediction_error=0.0, timestamp=0.0)
    store.add_node(a)
    with pytest.raises(KeyError):
        store.add_edge(Edge(source_id=a.node_id, target_id="ghost", edge_type=EdgeType.RESULT))


def test_delete_node_removes_node_and_incident_edges():
    store = FaissNetworkXStore()
    a = MemoryNode(state_vector=[1.0, 0.0], prediction_error=0.0, timestamp=0.0)
    b = MemoryNode(state_vector=[0.0, 1.0], prediction_error=0.0, timestamp=1.0)
    store.add_node(a)
    store.add_node(b)
    store.add_edge(Edge(source_id=a.node_id, target_id=b.node_id, edge_type=EdgeType.RESULT))

    assert store.delete_node(b.node_id) is True
    assert store.get_node(b.node_id) is None
    assert store.get_edges(a.node_id) == []
    assert store.delete_node(b.node_id) is False


def test_delete_removes_vector_from_search():
    """A deleted node must never come back from similarity search."""
    store = FaissNetworkXStore()
    a = MemoryNode(state_vector=[1.0, 0.0], prediction_error=0.0, timestamp=0.0)
    b = MemoryNode(state_vector=[0.9, 0.1], prediction_error=0.0, timestamp=1.0)
    store.add_node(a)
    store.add_node(b)
    store.delete_node(a.node_id)
    hits = store.query_similar(np.array([1.0, 0.0], dtype=np.float32), k=10)
    assert [h[0] for h in hits] == [b.node_id]


def test_readd_overwrites_node():
    store = FaissNetworkXStore()
    a = MemoryNode(state_vector=[1.0, 0.0], prediction_error=0.0, timestamp=0.0)
    store.add_node(a)
    updated = MemoryNode(
        state_vector=[0.0, 1.0],
        prediction_error=0.9,
        timestamp=2.0,
        node_id=a.node_id,
    )
    store.add_node(updated)
    assert store.node_count() == 1
    fetched = store.get_node(a.node_id)
    assert fetched.prediction_error == 0.9
    np.testing.assert_array_equal(fetched.state_vector, np.array([0.0, 1.0], dtype=np.float32))


def test_query_similar_ignores_action_nodes():
    store = FaissNetworkXStore()
    mem = MemoryNode(state_vector=[1.0, 0.0], prediction_error=0.0, timestamp=0.0)
    act = ActionNode(action_vector=[1.0, 0.0], action_type="x")
    store.add_node(mem)
    store.add_node(act)
    hits = store.query_similar(np.array([1.0, 0.0], dtype=np.float32), k=10)
    assert [h[0] for h in hits] == [mem.node_id]


def test_query_similar_k_zero_returns_empty(faiss_memory):
    assert faiss_memory.store.query_similar(faiss_memory.cluster_centers[0], 0) == []


def test_query_empty_store_returns_empty():
    assert FaissNetworkXStore().query_similar(np.array([1.0, 0.0], dtype=np.float32), 5) == []


# --------------------------------------------------------------------------- #
# vector search structure (TODO 1.8 tests 1 & 2)
# --------------------------------------------------------------------------- #


def test_generated_store_has_expected_node_count(faiss_memory):
    assert faiss_memory.store.node_count() == N_NODES * 2


def test_query_near_cluster_returns_that_cluster(faiss_memory):
    store = faiss_memory.store
    cluster_idx = 2
    center = faiss_memory.cluster_centers[cluster_idx]
    expected = set(faiss_memory.cluster_node_ids[cluster_idx])

    hits = store.query_similar(center, k=10)
    returned = {node_id for node_id, _ in hits}
    assert returned <= expected
    assert max(dist for _, dist in hits) < 0.05


def test_query_empty_space_returns_distant(faiss_memory):
    store = faiss_memory.store
    far = faiss_memory.empty_region_vector()
    hits = store.query_similar(far, k=5)
    assert min(dist for _, dist in hits) > 0.5


def test_query_sorted_by_ascending_distance(faiss_memory):
    hits = faiss_memory.store.query_similar(faiss_memory.cluster_centers[0], k=10)
    dists = [d for _, d in hits]
    assert dists == sorted(dists)


def test_hnsw_recall_matches_bruteforce(faiss_memory):
    """HNSW recall@10 must be high vs exact brute force at POC scale.

    The exact reference is built from the *same* node objects (ids are random
    UUIDs, so a separately generated store shares none). Compared by set overlap,
    not order: cluster members sit at near-tie distances and can rank slightly
    differently while returning the same neighbourhood.
    """
    from src.memory import InMemoryStore

    # Brute-force reference over the same cluster members (background nodes are
    # far from any cluster centre and never enter a top-10 there).
    exact = InMemoryStore()
    for ids in faiss_memory.cluster_node_ids:
        for nid in ids:
            exact.add_node(faiss_memory.store.get_node(nid))

    for center in faiss_memory.cluster_centers:
        hnsw_ids = {nid for nid, _ in faiss_memory.store.query_similar(center, k=10)}
        exact_ids = {nid for nid, _ in exact.query_similar(center, k=10)}
        recall = len(hnsw_ids & exact_ids) / len(exact_ids)
        assert recall >= 0.8


# --------------------------------------------------------------------------- #
# retrieval + edge chains (TODO 1.8 test 5)
# --------------------------------------------------------------------------- #


def test_retriever_expands_action_and_result_edges(faiss_memory):
    retriever = Retriever(faiss_memory.store)
    result = retriever.retrieve(faiss_memory.cluster_centers[1], k=5)

    assert isinstance(result, RetrievalResult)
    assert result.count == 5
    assert all(m.action is not None for m in result.memories)
    assert all(isinstance(m.action, ActionNode) for m in result.memories)
    assert any(m.result_state is not None for m in result.memories)


def test_edge_chain_follows_result_edges():
    store = FaissNetworkXStore()
    nodes = [
        MemoryNode(state_vector=[float(i)], prediction_error=0.0, timestamp=float(i))
        for i in range(4)
    ]
    for n in nodes:
        store.add_node(n)
    for a, b in zip(nodes, nodes[1:]):
        store.add_edge(Edge(source_id=a.node_id, target_id=b.node_id, edge_type=EdgeType.RESULT))

    chain = store.get_edge_chain(nodes[0].node_id, hops=10)
    assert [n.node_id for n in chain] == [n.node_id for n in nodes[1:]]
    assert len(store.get_edge_chain(nodes[0].node_id, hops=2)) == 2


def test_edge_chain_handles_cycles():
    store = FaissNetworkXStore()
    a = MemoryNode(state_vector=[0.0], prediction_error=0.0, timestamp=0.0)
    b = MemoryNode(state_vector=[1.0], prediction_error=0.0, timestamp=1.0)
    store.add_node(a)
    store.add_node(b)
    store.add_edge(Edge(source_id=a.node_id, target_id=b.node_id, edge_type=EdgeType.RESULT))
    store.add_edge(Edge(source_id=b.node_id, target_id=a.node_id, edge_type=EdgeType.RESULT))
    chain = store.get_edge_chain(a.node_id, hops=100)
    assert [n.node_id for n in chain] == [b.node_id]


def test_get_edges_filters_by_type(faiss_memory):
    store = faiss_memory.store
    node_id = faiss_memory.cluster_node_ids[0][0]
    assert len(store.get_edges(node_id, EdgeType.ACTION)) == 1
    assert len(store.get_edges(node_id, EdgeType.RESULT)) == 1
    assert len(store.get_edges(node_id)) == 2


# --------------------------------------------------------------------------- #
# confidence + novelty (TODO 1.8 tests 3 & 4)
# --------------------------------------------------------------------------- #


def test_confidence_high_for_cluster_low_for_empty(faiss_memory):
    retriever = Retriever(faiss_memory.store)
    conf = SimpleConfidence()
    near = retriever.retrieve(faiss_memory.cluster_centers[0], k=5)
    far = retriever.retrieve(faiss_memory.empty_region_vector(), k=5)
    conf_near = conf.compute(near)
    conf_far = conf.compute(far)
    assert conf_near > 0.5
    assert conf_far < 0.2
    assert conf_near > conf_far


def test_novelty_low_for_cluster_high_for_empty(faiss_memory):
    retriever = Retriever(faiss_memory.store)
    nov = SimpleNovelty()
    near = retriever.retrieve(faiss_memory.cluster_centers[0], k=5)
    far = retriever.retrieve(faiss_memory.empty_region_vector(), k=5)
    nov_near = nov.compute(near)
    nov_far = nov.compute(far)
    assert nov_near < 0.3
    assert nov_far > 0.5
    assert nov_far > nov_near


# --------------------------------------------------------------------------- #
# learning progress (TODO 1.9)
# --------------------------------------------------------------------------- #


def _cluster_result_with_errors(errors, dim=DIM):
    store = FaissNetworkXStore()
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    for i, err in enumerate(errors):
        store.add_node(
            MemoryNode(state_vector=base.copy(), prediction_error=float(err), timestamp=float(i))
        )
    return Retriever(store).retrieve(base, k=len(errors))


def test_learning_progress_negative_for_decreasing_error():
    errors = list(np.linspace(1.0, 0.1, 50))
    slope = LinearLearningProgress().compute(_cluster_result_with_errors(errors))
    assert slope < -0.001


def test_learning_progress_flat_for_constant_error():
    errors = [0.5] * 50
    slope = LinearLearningProgress().compute(_cluster_result_with_errors(errors))
    assert abs(slope) < 1e-6


# --------------------------------------------------------------------------- #
# persistence (TODO 1.8 test 6)
# --------------------------------------------------------------------------- #


def test_persist_and_reload_restores_everything(faiss_memory, tmp_path):
    store = faiss_memory.store
    path = str(tmp_path / "memory.pkl")
    store.persist(path)

    reloaded = FaissNetworkXStore()
    reloaded.load(path)

    assert reloaded.node_count() == store.node_count()

    sample_id = faiss_memory.cluster_node_ids[0][0]
    original = store.get_node(sample_id)
    restored = reloaded.get_node(sample_id)
    assert restored is not None
    np.testing.assert_array_equal(restored.state_vector, original.state_vector)
    assert restored.prediction_error == original.prediction_error

    center = faiss_memory.cluster_centers[0]
    orig_hits = store.query_similar(center, k=5)
    new_hits = reloaded.query_similar(center, k=5)
    assert [h[0] for h in orig_hits] == [h[0] for h in new_hits]
    assert reloaded.get_edges(sample_id, EdgeType.ACTION)


def test_persist_is_atomic_via_tmp(tmp_path):
    store = FaissNetworkXStore()
    n = MemoryNode(state_vector=[1.0], prediction_error=0.0, timestamp=0.0)
    store.add_node(n)
    path = str(tmp_path / "nested" / "deep" / "mem.pkl")  # dirs don't exist yet
    store.persist(path)
    reloaded = FaissNetworkXStore()
    reloaded.load(path)
    assert reloaded.node_count() == 1


def test_rebuild_index_preserves_search(faiss_memory):
    """Explicit rebuild must not change query results."""
    store = faiss_memory.store
    center = faiss_memory.cluster_centers[3]
    before = store.query_similar(center, k=10)
    store.rebuild_index()
    after = store.query_similar(center, k=10)
    assert [h[0] for h in before] == [h[0] for h in after]
