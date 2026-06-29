"""Prune / sleep cycle for bounding memory growth (POC 1, TODO 1.11).

At 30 fps the graph gains ~2.6M nodes/day — unbounded, it dies in hours. The
drive's own logic says which memories are worth keeping (PROJECT_CONTEXT
"Memory Growth and Pruning"):

* **KEEP** high-prediction-error nodes — surprising moments are informative
  regardless of how often they recur.
* **KEEP** highly connected nodes — junctions where many experiences meet.
* **PRUNE** low-error *and* low-connectivity nodes — boring and already
  well-predicted, safe to forget.

This module implements that pruning rule (the deletion half of TODO 1.11).
**Skill compression** — folding frequent low-error chains into prototype "skill"
nodes — is listed as genuinely unsolved in PROJECT_CONTEXT and is deliberately
left out here; this is the basic, safe prune only.

Trigger: :func:`should_sleep` fires when the store crosses ``sleep_threshold``.
The loop then pauses collection, calls :func:`prune`, persists, and resumes —
:func:`run_sleep_cycle` wires that to an :class:`~src.memory.persistence.AutoPersister`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .faiss_networkx_store import FaissNetworkXStore


@dataclass
class PruneConfig:
    """Thresholds for the sleep/prune cycle (TODO 1.11).

    Attributes:
        sleep_threshold: Node count at which :func:`should_sleep` fires.
        error_keep_threshold: Prediction error at/above which a node is always
            kept (surprising => informative).
        degree_keep_threshold: Total degree at/above which a node is always kept
            (a junction in experience).
        target_count: Optional hard ceiling. If pruning the boring nodes still
            leaves more than this, the least-valuable *protected* nodes (lowest
            error, then lowest degree) are dropped too until the ceiling is met.
            ``None`` means "delete only the boring nodes, no hard cap".
    """

    sleep_threshold: int = 50_000
    error_keep_threshold: float = 0.5
    degree_keep_threshold: int = 2
    target_count: Optional[int] = None


@dataclass
class PruneStats:
    """Outcome of a :func:`prune` pass.

    Attributes:
        nodes_before: Total node count before pruning.
        nodes_after: Total node count after pruning.
        memory_pruned: ``MemoryNode``s deleted.
        actions_pruned: orphaned ``ActionNode``s deleted (their state node went).
    """

    nodes_before: int
    nodes_after: int
    memory_pruned: int
    actions_pruned: int

    @property
    def total_pruned(self) -> int:
        """Total nodes removed (memory + orphaned actions)."""
        return self.memory_pruned + self.actions_pruned


def should_sleep(store: FaissNetworkXStore, config: PruneConfig = PruneConfig()) -> bool:
    """True when the store has grown to/past the sleep threshold (TODO 1.11)."""
    return store.node_count() >= config.sleep_threshold


def prune(
    store: FaissNetworkXStore, config: PruneConfig = PruneConfig()
) -> PruneStats:
    """Delete boring memories, keep surprising/connected ones (TODO 1.11).

    Rule: a ``MemoryNode`` survives if its prediction error is high
    (``>= error_keep_threshold``) *or* it is well connected
    (``degree >= degree_keep_threshold``); otherwise it is pruned. If
    ``config.target_count`` is set and the survivors still exceed it, the
    least-valuable protected nodes are dropped too until the ceiling is met.

    ``ActionNode``s left orphaned by their state node's deletion are swept up.
    The FAISS index is rebuilt once at the end (HNSW can't delete in place), so
    the store is immediately queryable again on wake.
    """
    before = store.node_count()

    # Score every memory node; remember error/degree for the optional hard cap.
    boring: List[str] = []
    protected: List[Tuple[float, int, str]] = []  # (error, degree, id)
    for nid in store.memory_node_ids():
        node = store.get_node(nid)
        degree = store.degree(nid)
        keep = (
            node.prediction_error >= config.error_keep_threshold
            or degree >= config.degree_keep_threshold
        )
        if keep:
            protected.append((node.prediction_error, degree, nid))
        else:
            boring.append(nid)

    to_delete = list(boring)

    # Optional hard ceiling: if dropping the boring nodes isn't enough, give up
    # the least-valuable protected nodes too (lowest error, then lowest degree).
    if config.target_count is not None:
        projected = before - len(to_delete)
        overflow = projected - config.target_count
        if overflow > 0:
            protected.sort(key=lambda t: (t[0], t[1]))  # least valuable first
            to_delete.extend(nid for _, _, nid in protected[:overflow])

    # Stage 1: delete the chosen memory nodes (defer the index rebuild).
    memory_pruned = store.delete_nodes(to_delete, rebuild=False)

    # Stage 2: sweep up action nodes orphaned by those deletions.
    orphans = [aid for aid in store.action_node_ids() if store.degree(aid) == 0]
    actions_pruned = store.delete_nodes(orphans, rebuild=False)

    # One rebuild to clear all tombstones and make the store queryable again.
    store.rebuild_index()

    return PruneStats(
        nodes_before=before,
        nodes_after=store.node_count(),
        memory_pruned=memory_pruned,
        actions_pruned=actions_pruned,
    )


def run_sleep_cycle(persister, config: PruneConfig = PruneConfig()) -> PruneStats:
    """Prune the persister's store then snapshot it (TODO 1.11 steps 4-5).

    Convenience that ties the prune cycle to auto-persistence: prune, then force
    an atomic snapshot of the pruned graph + rebuilt index so the bounded state
    is durable before collection resumes. ``persister.store`` must be a
    :class:`FaissNetworkXStore`.
    """
    stats = prune(persister.store, config)
    persister.persist_now()
    return stats
