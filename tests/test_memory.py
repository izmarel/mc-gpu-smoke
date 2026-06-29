"""POC 1 memory tests (TODO 1.8 + 1.9).

Exercises the full stack against the dict-based reference backend:

* models — validation and round-trip serialisation;
* storage — CRUD, vector search, edge traversal, persistence;
* retrieval — neighbourhood assembly with ACTION/RESULT edges;
* signals — confidence, novelty, learning progress, on both clustered (lots of
  precedent) and empty (novel) regions of vector space.

Run with: ``.venv/bin/pytest tests/ -v``
"""

from __future__ import annotations

import numpy as np
import pytest

from src.memory import (
    ActionNode,
    Edge,
    EdgeType,
    InMemoryStore,
    LinearLearningProgress,
    MemoryNode,
    RetrievalResult,
    Retriever,
    SimpleConfidence,
    SimpleNovelty,
    node_from_dict,
)
from src.memory.storage import cosine_distance, l2_distance

from conftest import DIM, N_NODES, generate_fake_memory


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


def test_memory_node_coerces_to_float32_1d():
    node = MemoryNode(state_vector=[1, 2, 3], prediction_error=0.5, timestamp=10)
    assert node.state_vector.dtype == np.float32
    assert node.state_vector.ndim == 1
    assert node.dim == 3
    assert isinstance(node.prediction_error, float)


def test_memory_node_rejects_2d_vector():
    with pytest.raises(ValueError):
        MemoryNode(state_vector=np.zeros((2, 2)), prediction_error=0.0, timestamp=0.0)


def test_memory_node_rejects_empty_vector():
    with pytest.raises(ValueError):
        MemoryNode(state_vector=[], prediction_error=0.0, timestamp=0.0)


def test_node_ids_are_unique():
    a = MemoryNode(state_vector=[1.0], prediction_error=0.0, timestamp=0.0)
    b = MemoryNode(state_vector=[1.0], prediction_error=0.0, timestamp=0.0)
    assert a.node_id != b.node_id


def test_edge_coerces_string_type():
    edge = Edge(source_id="a", target_id="b", edge_type="ACTION")
    assert edge.edge_type is EdgeType.ACTION


def test_memory_node_roundtrip():
    node = MemoryNode(
        state_vector=[1.0, 2.0, 3.0],
        prediction_error=0.7,
        timestamp=123.0,
        text_metadata="hello",
    )
    restored = node_from_dict(node.to_dict())
    assert isinstance(restored, MemoryNode)
    assert restored.node_id == node.node_id
    assert restored.text_metadata == "hello"
    np.testing.assert_array_equal(restored.state_vector, node.state_vector)


def test_action_node_roundtrip():
    action = ActionNode(action_vector=[0.1, 0.2], action_type="click")
    restored = node_from_dict(action.to_dict())
    assert isinstance(restored, ActionNode)
    assert restored.action_type == "click"
    np.testing.assert_array_equal(restored.action_vector, action.action_vector)


# --------------------------------------------------------------------------- #
# distance helpers
# --------------------------------------------------------------------------- #


def test_cosine_distance_identical_is_zero():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_opposite_is_two():
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_distance(v, -v) == pytest.approx(2.0, abs=1e-6)


def test_cosine_distance_zero_vector_safe():
    assert cosine_distance(np.zeros(3), np.ones(3)) == 1.0


def test_l2_distance_basic():
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    assert l2_distance(a, b) == pytest.approx(25.0)


# --------------------------------------------------------------------------- #
# storage: CRUD
# --------------------------------------------------------------------------- #


def test_add_and_get_node():
    store = InMemoryStore()
    node = MemoryNode(state_vector=[1.0, 0.0], prediction_error=0.1, timestamp=1.0)
    store.add_node(node)
    fetched = store.get_node(node.node_id)
    assert fetched is node
    assert store.node_count() == 1
    assert len(store) == 1


def test_get_missing_node_returns_none():
    assert InMemoryStore().get_node("nope") is None


def test_add_edge_requires_existing_endpoints():
    store = InMemoryStore()
    a = MemoryNode(state_vector=[1.0], prediction_error=0.0, timestamp=0.0)
    store.add_node(a)
    with pytest.raises(KeyError):
        store.add_edge(Edge(source_id=a.node_id, target_id="ghost", edge_type=EdgeType.RESULT))


def test_delete_node_removes_node_and_incident_edges():
    store = InMemoryStore()
    a = MemoryNode(state_vector=[1.0, 0.0], prediction_error=0.0, timestamp=0.0)
    b = MemoryNode(state_vector=[0.0, 1.0], prediction_error=0.0, timestamp=1.0)
    store.add_node(a)
    store.add_node(b)
    store.add_edge(Edge(source_id=a.node_id, target_id=b.node_id, edge_type=EdgeType.RESULT))

    assert store.delete_node(b.node_id) is True
    assert store.get_node(b.node_id) is None
    # The incoming edge from a -> b must be gone too.
    assert store.get_edges(a.node_id) == []
    # Deleting again is a no-op.
    assert store.delete_node(b.node_id) is False


def test_query_similar_ignores_action_nodes():
    store = InMemoryStore()
    mem = MemoryNode(state_vector=[1.0, 0.0], prediction_error=0.0, timestamp=0.0)
    act = ActionNode(action_vector=[1.0, 0.0], action_type="x")
    store.add_node(mem)
    store.add_node(act)
    hits = store.query_similar(np.array([1.0, 0.0], dtype=np.float32), k=10)
    assert [h[0] for h in hits] == [mem.node_id]


def test_query_similar_k_zero_returns_empty(fake_memory):
    assert fake_memory.store.query_similar(fake_memory.cluster_centers[0], 0) == []


# --------------------------------------------------------------------------- #
# storage: vector search structure (TODO 1.8 tests 1 & 2)
# --------------------------------------------------------------------------- #


def test_generated_store_has_expected_node_count(fake_memory):
    # N_NODES memory nodes + one action node each.
    assert fake_memory.store.node_count() == N_NODES * 2


def test_query_near_cluster_returns_that_cluster(fake_memory):
    """TODO 1.8 test 1: a query near a cluster centre returns its members."""
    store = fake_memory.store
    cluster_idx = 2
    center = fake_memory.cluster_centers[cluster_idx]
    expected = set(fake_memory.cluster_node_ids[cluster_idx])

    hits = store.query_similar(center, k=10)
    returned = {node_id for node_id, _ in hits}

    # Every one of the 10 nearest should belong to the queried cluster.
    assert returned <= expected
    # ...and they should be genuinely close.
    assert max(dist for _, dist in hits) < 0.05


def test_query_empty_space_returns_distant(fake_memory):
    """TODO 1.8 test 2: a query in empty space returns only distant matches."""
    store = fake_memory.store
    far = fake_memory.empty_region_vector()
    hits = store.query_similar(far, k=5)
    # Something always comes back (brute force), but it must be far away —
    # much farther than the intra-cluster distances above.
    assert min(dist for _, dist in hits) > 0.5


# --------------------------------------------------------------------------- #
# retrieval + edge chains (TODO 1.8 test 5)
# --------------------------------------------------------------------------- #


def test_retriever_expands_action_and_result_edges(fake_memory):
    retriever = Retriever(fake_memory.store)
    cluster_idx = 1
    result = retriever.retrieve(fake_memory.cluster_centers[cluster_idx], k=5)

    assert isinstance(result, RetrievalResult)
    assert result.count == 5
    # Every cluster node has an ACTION edge; all but the last in a chain have a
    # RESULT edge.
    assert all(m.action is not None for m in result.memories)
    assert all(isinstance(m.action, ActionNode) for m in result.memories)
    assert any(m.result_state is not None for m in result.memories)


def test_edge_chain_follows_result_edges():
    """TODO 1.8 test 5: RESULT edges form valid state->state chains."""
    store = InMemoryStore()
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

    # Bounded by hops.
    assert len(store.get_edge_chain(nodes[0].node_id, hops=2)) == 2


def test_edge_chain_handles_cycles():
    store = InMemoryStore()
    a = MemoryNode(state_vector=[0.0], prediction_error=0.0, timestamp=0.0)
    b = MemoryNode(state_vector=[1.0], prediction_error=0.0, timestamp=1.0)
    store.add_node(a)
    store.add_node(b)
    store.add_edge(Edge(source_id=a.node_id, target_id=b.node_id, edge_type=EdgeType.RESULT))
    store.add_edge(Edge(source_id=b.node_id, target_id=a.node_id, edge_type=EdgeType.RESULT))
    # Must terminate, not loop forever. The start node is never revisited, so the
    # chain is just [b] (a -> b, then b -> a is pruned as already seen).
    chain = store.get_edge_chain(a.node_id, hops=100)
    assert [n.node_id for n in chain] == [b.node_id]


def test_get_edges_filters_by_type(fake_memory):
    store = fake_memory.store
    # A non-terminal cluster node has both an ACTION and a RESULT edge.
    node_id = fake_memory.cluster_node_ids[0][0]
    assert len(store.get_edges(node_id, EdgeType.ACTION)) == 1
    assert len(store.get_edges(node_id, EdgeType.RESULT)) == 1
    assert len(store.get_edges(node_id)) == 2


# --------------------------------------------------------------------------- #
# confidence + novelty (TODO 1.8 tests 3 & 4)
# --------------------------------------------------------------------------- #


def test_confidence_high_for_cluster_low_for_empty(fake_memory):
    """TODO 1.8 test 3: confidence is high near a cluster, ~0 in empty space."""
    retriever = Retriever(fake_memory.store)
    conf = SimpleConfidence()

    near = retriever.retrieve(fake_memory.cluster_centers[0], k=5)
    far = retriever.retrieve(fake_memory.empty_region_vector(), k=5)

    conf_near = conf.compute(near)
    conf_far = conf.compute(far)

    assert conf_near > 0.5
    assert conf_far < 0.2
    assert conf_near > conf_far


def test_confidence_zero_for_empty_result():
    assert SimpleConfidence().compute(RetrievalResult(query_vector=np.zeros(2))) == 0.0


def test_confidence_in_unit_range(fake_memory):
    retriever = Retriever(fake_memory.store)
    conf = SimpleConfidence()
    for c in range(len(fake_memory.cluster_centers)):
        val = conf.compute(retriever.retrieve(fake_memory.cluster_centers[c], k=5))
        assert 0.0 <= val <= 1.0


def test_novelty_low_for_cluster_high_for_empty(fake_memory):
    """TODO 1.8 test 4: novelty is low near a cluster, high in empty space."""
    retriever = Retriever(fake_memory.store)
    nov = SimpleNovelty()

    near = retriever.retrieve(fake_memory.cluster_centers[0], k=5)
    far = retriever.retrieve(fake_memory.empty_region_vector(), k=5)

    nov_near = nov.compute(near)
    nov_far = nov.compute(far)

    assert nov_near < 0.3
    assert nov_far > 0.5
    assert nov_far > nov_near


def test_novelty_max_for_empty_result():
    assert SimpleNovelty().compute(RetrievalResult(query_vector=np.zeros(2))) == 1.0


# --------------------------------------------------------------------------- #
# persistence (TODO 1.8 test 6)
# --------------------------------------------------------------------------- #


def test_persist_and_reload_restores_everything(fake_memory, tmp_path):
    """TODO 1.8 test 6: persist to disk, reload, everything matches."""
    store = fake_memory.store
    path = str(tmp_path / "memory.json")
    store.persist(path)

    reloaded = InMemoryStore()
    reloaded.load(path)

    assert reloaded.node_count() == store.node_count()

    # Spot-check a node round-trips exactly.
    sample_id = fake_memory.cluster_node_ids[0][0]
    original = store.get_node(sample_id)
    restored = reloaded.get_node(sample_id)
    assert restored is not None
    np.testing.assert_array_equal(restored.state_vector, original.state_vector)
    assert restored.prediction_error == original.prediction_error

    # Edges survive: querying + chains behave identically.
    center = fake_memory.cluster_centers[0]
    orig_hits = store.query_similar(center, k=5)
    new_hits = reloaded.query_similar(center, k=5)
    assert [h[0] for h in orig_hits] == [h[0] for h in new_hits]
    assert reloaded.get_edges(sample_id, EdgeType.ACTION)


def test_persist_is_atomic_via_tmp(tmp_path):
    store = InMemoryStore()
    n = MemoryNode(state_vector=[1.0], prediction_error=0.0, timestamp=0.0)
    store.add_node(n)
    path = str(tmp_path / "nested" / "deep" / "mem.json")  # dirs don't exist yet
    store.persist(path)  # must create parent dirs
    reloaded = InMemoryStore()
    reloaded.load(path)
    assert reloaded.node_count() == 1


# --------------------------------------------------------------------------- #
# learning progress (TODO 1.9)
# --------------------------------------------------------------------------- #


def _cluster_result_with_errors(errors, dim=DIM):
    """Build a RetrievalResult over a single cluster with the given error series
    at increasing timestamps."""
    store = InMemoryStore()
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    ids = []
    for i, err in enumerate(errors):
        node = MemoryNode(
            state_vector=base.copy(),
            prediction_error=float(err),
            timestamp=float(i),
        )
        store.add_node(node)
        ids.append(node.node_id)
    return Retriever(store).retrieve(base, k=len(errors))


def test_learning_progress_negative_for_decreasing_error():
    """TODO 1.9: errors trending down over time => negative slope (learning)."""
    errors = list(np.linspace(1.0, 0.1, 50))
    result = _cluster_result_with_errors(errors)
    slope = LinearLearningProgress().compute(result)
    assert slope < -0.001


def test_learning_progress_flat_for_constant_error():
    """TODO 1.9: flat errors => slope ~ 0 (the noisy-TV case)."""
    errors = [0.5] * 50
    result = _cluster_result_with_errors(errors)
    slope = LinearLearningProgress().compute(result)
    assert abs(slope) < 1e-6


def test_learning_progress_positive_for_increasing_error():
    errors = list(np.linspace(0.1, 1.0, 50))
    result = _cluster_result_with_errors(errors)
    slope = LinearLearningProgress().compute(result)
    assert slope > 0.001


def test_learning_progress_needs_two_points():
    result = _cluster_result_with_errors([0.5])
    assert LinearLearningProgress().compute(result) == 0.0


def test_learning_progress_handles_identical_timestamps():
    store = InMemoryStore()
    base = np.array([1.0, 0.0], dtype=np.float32)
    for err in (0.1, 0.9):
        store.add_node(
            MemoryNode(state_vector=base.copy(), prediction_error=err, timestamp=5.0)
        )
    result = Retriever(store).retrieve(base, k=2)
    assert LinearLearningProgress().compute(result) == 0.0


# --------------------------------------------------------------------------- #
# protocol conformance
# --------------------------------------------------------------------------- #


def test_concrete_computers_satisfy_protocols():
    from src.memory import (
        ConfidenceComputer,
        LearningProgressComputer,
        NoveltyComputer,
    )

    assert isinstance(SimpleConfidence(), ConfidenceComputer)
    assert isinstance(SimpleNovelty(), NoveltyComputer)
    assert isinstance(LinearLearningProgress(), LearningProgressComputer)


def test_generator_is_deterministic():
    a = generate_fake_memory(n_nodes=50, seed=7)
    b = generate_fake_memory(n_nodes=50, seed=7)
    np.testing.assert_array_equal(a.cluster_centers, b.cluster_centers)
