"""Prune / sleep-cycle tests (TODO 1.11).

The acceptance test builds a 60k-node graph mixing boring (low-error, low-degree)
nodes, surprising (high-error) nodes, and junctions (high-degree), triggers a
prune, and checks the bound holds: count drops, every high-error node survives,
junctions survive, and the graph is still traversable + queryable afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.memory import (
    AutoPersister,
    Edge,
    EdgeType,
    FaissNetworkXStore,
    MemoryNode,
    PruneConfig,
    Retriever,
    prune,
    run_sleep_cycle,
    should_sleep,
)

DIM = 8


def _vec(rng) -> np.ndarray:
    return rng.standard_normal(DIM).astype(np.float32)


def _build_mixed_store(rng, n_boring, n_surprising, n_junctions):
    """A store with boring (prunable), surprising (keep), and junction (keep) nodes.

    Returns ``(store, surprising_ids, junction_ids)``.
    """
    store = FaissNetworkXStore()

    # Boring: low error, no edges -> prunable.
    for _ in range(n_boring):
        store.add_node(MemoryNode(state_vector=_vec(rng), prediction_error=0.05, timestamp=0.0))

    # Surprising: high error, no edges -> kept by the error rule.
    surprising_ids = []
    for _ in range(n_surprising):
        node = MemoryNode(state_vector=_vec(rng), prediction_error=0.95, timestamp=0.0)
        store.add_node(node)
        surprising_ids.append(node.node_id)

    # Junctions: low error but many incident edges -> kept by the degree rule.
    # Each is the RESULT target of several (low-error) feeder nodes.
    junction_ids = []
    for _ in range(n_junctions):
        hub = MemoryNode(state_vector=_vec(rng), prediction_error=0.05, timestamp=0.0)
        store.add_node(hub)
        junction_ids.append(hub.node_id)
        for _ in range(3):
            # High-error feeders: survive pruning so junctions keep degree >= 2.
            feeder = MemoryNode(state_vector=_vec(rng), prediction_error=0.95, timestamp=0.0)
            store.add_node(feeder)
            store.add_edge(
                Edge(source_id=feeder.node_id, target_id=hub.node_id, edge_type=EdgeType.RESULT)
            )
    return store, surprising_ids, junction_ids


# --------------------------------------------------------------------------- #
# trigger
# --------------------------------------------------------------------------- #


def test_should_sleep_fires_at_threshold():
    store = FaissNetworkXStore()
    cfg = PruneConfig(sleep_threshold=3)
    assert not should_sleep(store, cfg)
    for i in range(3):
        store.add_node(MemoryNode(state_vector=[float(i)], prediction_error=0.0, timestamp=0.0))
    assert should_sleep(store, cfg)


# --------------------------------------------------------------------------- #
# rule behaviour
# --------------------------------------------------------------------------- #


def test_prune_keeps_high_error_and_junctions():
    rng = np.random.default_rng(0)
    store, surprising_ids, junction_ids = _build_mixed_store(
        rng, n_boring=200, n_surprising=50, n_junctions=10
    )
    before = store.node_count()

    stats = prune(store, PruneConfig(error_keep_threshold=0.5, degree_keep_threshold=2))

    assert stats.nodes_after < before
    assert stats.memory_pruned > 0
    # Every surprising node survives.
    for nid in surprising_ids:
        assert store.get_node(nid) is not None
    # Every junction survives (degree >= 2).
    for nid in junction_ids:
        assert store.get_node(nid) is not None


def test_prune_target_count_enforces_hard_cap():
    rng = np.random.default_rng(1)
    store, _, _ = _build_mixed_store(rng, n_boring=100, n_surprising=200, n_junctions=0)
    # 200 surprising nodes are all rule-protected, but the cap forces deletions.
    stats = prune(store, PruneConfig(target_count=50))
    assert store.node_count() <= 50
    assert stats.nodes_after <= 50


def test_prune_store_still_queryable_and_traversable():
    rng = np.random.default_rng(2)
    store, surprising_ids, junction_ids = _build_mixed_store(
        rng, n_boring=300, n_surprising=40, n_junctions=10
    )
    prune(store, PruneConfig())

    # Index rebuilt: a query against a surviving vector still returns it.
    survivor = store.get_node(surprising_ids[0])
    hits = store.query_similar(survivor.state_vector, k=1)
    assert hits and hits[0][0] == surprising_ids[0]

    # Junctions still have their incident RESULT edges (graph intact).
    assert any(store.degree(jid) >= 2 for jid in junction_ids)


def test_run_sleep_cycle_persists_pruned_graph(tmp_path):
    rng = np.random.default_rng(3)
    store, surprising_ids, _ = _build_mixed_store(
        rng, n_boring=100, n_surprising=20, n_junctions=0
    )
    path = str(tmp_path / "sleep.pkl")
    persister = AutoPersister(store, path, interval=10_000)

    stats = run_sleep_cycle(persister, PruneConfig())
    assert stats.memory_pruned > 0

    reloaded = AutoPersister.open(path)
    assert reloaded.store.node_count() == stats.nodes_after
    for nid in surprising_ids:
        assert reloaded.store.get_node(nid) is not None


# --------------------------------------------------------------------------- #
# scale acceptance (TODO 1.11: 60k nodes)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_prune_at_60k_nodes():
    rng = np.random.default_rng(42)
    # 51k boring + 5k surprising + 1k junctions * 4 nodes (hub + 3 feeders) = 60k
    store, surprising_ids, junction_ids = _build_mixed_store(
        rng, n_boring=51_000, n_surprising=5_000, n_junctions=1_000
    )
    before = store.node_count()
    assert before >= 60_000

    stats = prune(store, PruneConfig(error_keep_threshold=0.5, degree_keep_threshold=2))

    assert stats.nodes_after < before
    assert stats.memory_pruned >= 50_000  # all boring leaf nodes gone
    # All 5k surprising survive.
    assert all(store.get_node(nid) is not None for nid in surprising_ids)
    # All 1k junctions survive and stay traversable.
    assert all(store.get_node(jid) is not None for jid in junction_ids)
    # Still queryable after the rebuild.
    survivor = store.get_node(surprising_ids[0])
    hits = store.query_similar(survivor.state_vector, k=1)
    assert hits and hits[0][0] == surprising_ids[0]
