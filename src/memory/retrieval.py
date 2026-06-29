"""Retrieval and derived signals (POC 1, TODO 1.4-1.7).

Everything the system "feels" about a situation falls out of one vector query
against memory. This module turns a raw similarity query into:

* :class:`RetrievalResult` — the structured neighbourhood (TODO 1.4): for each
  similar past state, the action taken, the state that resulted, and the
  surprise/time metadata.
* **confidence** (TODO 1.5): do we have enough consistent precedent to act?
* **novelty** (TODO 1.6): is this territory new?
* **learning progress** (TODO 1.7): is prediction error for this kind of state
  trending down over time (we're getting better) or flat (stuck)?

These are deliberately simple, transparent formulas — not neural nets — so their
behaviour is debuggable and matches the architecture's "signals fall out of
retrieval" principle. Each computer is a small class behind a
:class:`typing.Protocol` so it can be swapped (e.g. POC 4 replaces novelty with
world-model prediction-error trend) without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np

from .models import ActionNode, EdgeType, MemoryNode
from .storage import MemoryStore, SimilarityHit


@dataclass
class RetrievedMemory:
    """One similar past state and the experience hanging off it.

    Attributes:
        node: The matched ``MemoryNode``.
        distance: Distance from the query vector (smaller == more similar).
        action: ``ActionNode`` reached via the node's ACTION edge, if any.
        result_state: ``MemoryNode`` reached via the node's RESULT edge, if any
            (i.e. what happened next).
    """

    node: MemoryNode
    distance: float
    action: Optional[ActionNode] = None
    result_state: Optional[MemoryNode] = None

    @property
    def prediction_error(self) -> float:
        """Convenience accessor for the matched node's prediction error."""
        return self.node.prediction_error

    @property
    def timestamp(self) -> float:
        """Convenience accessor for the matched node's timestamp."""
        return self.node.timestamp


@dataclass
class RetrievalResult:
    """Structured neighbourhood returned for a query (TODO 1.4).

    Attributes:
        query_vector: The state vector that was searched with.
        memories: Similar past experiences, ordered most-similar first.
    """

    query_vector: np.ndarray
    memories: List[RetrievedMemory] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of similar memories retrieved."""
        return len(self.memories)

    @property
    def distances(self) -> List[float]:
        """Distances of the retrieved memories, most-similar first."""
        return [m.distance for m in self.memories]

    @property
    def prediction_errors(self) -> List[float]:
        """Prediction errors of the retrieved memories."""
        return [m.prediction_error for m in self.memories]

    @property
    def is_empty(self) -> bool:
        """True when nothing similar was found (completely novel territory)."""
        return not self.memories


class Retriever:
    """Turns a state vector into a :class:`RetrievalResult` (TODO 1.4).

    Runs the vector query against a :class:`MemoryStore`, then follows each
    hit's ACTION and RESULT edges to assemble the full experience around it.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(self, state_vector: np.ndarray, k: int = 5) -> RetrievalResult:
        """Retrieve up to ``k`` similar past states with their edges expanded."""
        hits: List[SimilarityHit] = self.store.query_similar(state_vector, k)
        memories: List[RetrievedMemory] = []
        for node_id, distance in hits:
            node = self.store.get_node(node_id)
            if not isinstance(node, MemoryNode):
                continue
            memories.append(
                RetrievedMemory(
                    node=node,
                    distance=distance,
                    action=self._first_target(node_id, EdgeType.ACTION, ActionNode),
                    result_state=self._first_target(
                        node_id, EdgeType.RESULT, MemoryNode
                    ),
                )
            )
        return RetrievalResult(
            query_vector=np.asarray(state_vector, dtype=np.float32),
            memories=memories,
        )

    def _first_target(self, node_id: str, edge_type: EdgeType, expected_type):
        """Return the target of the first edge of ``edge_type``, if it is of
        ``expected_type``."""
        edges = self.store.get_edges(node_id, edge_type)
        if not edges:
            return None
        target = self.store.get_node(edges[0].target_id)
        return target if isinstance(target, expected_type) else None


# --- derived-signal protocols -------------------------------------------------


@runtime_checkable
class ConfidenceComputer(Protocol):
    """Maps a retrieval neighbourhood to a confidence in ``[0, 1]`` (TODO 1.5)."""

    def compute(self, result: RetrievalResult) -> float: ...


@runtime_checkable
class NoveltyComputer(Protocol):
    """Maps a retrieval neighbourhood to a novelty in ``[0, 1]`` (TODO 1.6)."""

    def compute(self, result: RetrievalResult) -> float: ...


@runtime_checkable
class LearningProgressComputer(Protocol):
    """Maps a retrieval neighbourhood to a learning-progress signal (TODO 1.7).

    Output is a slope: negative == prediction error trending down over time
    (the system is getting better), ~0 == flat, positive == getting worse.
    """

    def compute(self, result: RetrievalResult) -> float: ...


# --- concrete implementations -------------------------------------------------


class SimpleConfidence:
    """Confidence from count, closeness and consistency of precedent (TODO 1.5).

    Three intuitions, multiplied together so any one being bad tanks confidence:

    * **support** — more similar memories => more confident. Saturates at
      ``saturation_count`` neighbours.
    * **closeness** — the closer those memories sit in vector space, the more
      they actually apply. Mapped from mean distance via ``distance_scale``.
    * **consistency** — if the precedents' prediction errors agree (low spread),
      the outcome is reliable; if they disagree, it isn't.

    No similar states => confidence 0.0 (completely novel, no basis to act).
    Many close, consistent precedents => confidence approaches 1.0.
    """

    def __init__(
        self,
        saturation_count: int = 5,
        distance_scale: float = 0.5,
        consistency_scale: float = 0.5,
    ) -> None:
        self.saturation_count = max(1, saturation_count)
        self.distance_scale = distance_scale
        self.consistency_scale = consistency_scale

    def compute(self, result: RetrievalResult) -> float:
        if result.is_empty:
            return 0.0

        support = min(1.0, result.count / self.saturation_count)

        mean_dist = float(np.mean(result.distances))
        closeness = float(np.exp(-mean_dist / self.distance_scale))

        errors = result.prediction_errors
        if len(errors) >= 2:
            spread = float(np.std(errors))
            consistency = float(np.exp(-spread / self.consistency_scale))
        else:
            # A single precedent: no spread to measure, give neutral weight.
            consistency = 0.5

        return float(max(0.0, min(1.0, support * closeness * consistency)))


class SimpleNovelty:
    """Novelty from how much (and how close) precedent exists (TODO 1.6).

    * No similar nodes => novelty 1.0 (brand-new territory).
    * Similar nodes but far away => moderate novelty.
    * Many close similar nodes => novelty approaching 0.0.

    Computed from the nearest neighbour's distance blended with how full the
    neighbourhood is relative to ``expected_count``.
    """

    def __init__(
        self, distance_scale: float = 0.5, expected_count: int = 5
    ) -> None:
        self.distance_scale = distance_scale
        self.expected_count = max(1, expected_count)

    def compute(self, result: RetrievalResult) -> float:
        if result.is_empty:
            return 1.0

        nearest = min(result.distances)
        # Distance term: 0 at identical match, -> 1 as nearest neighbour recedes.
        distance_novelty = 1.0 - float(np.exp(-nearest / self.distance_scale))

        # Sparsity term: a thin neighbourhood is more novel than a full one.
        sparsity = 1.0 - min(1.0, result.count / self.expected_count)

        # Weight distance more heavily — a single very-close match means low
        # novelty even if few neighbours were returned.
        novelty = 0.7 * distance_novelty + 0.3 * sparsity
        return float(max(0.0, min(1.0, novelty)))


class LinearLearningProgress:
    """Slope of prediction error vs. time for the retrieved cluster (TODO 1.7).

    Fits ``prediction_error = a * timestamp + b`` by ordinary least squares over
    the retrieved memories and returns the slope ``a``:

    * ``a < 0`` — error falling over time => learning (good).
    * ``a ~ 0`` — flat => not learning (the noisy-TV case stays here forever).
    * ``a > 0`` — error rising => getting worse.

    Timestamps are mean-centred before the fit for numerical stability with
    large unix values. Needs at least two distinct timestamps; otherwise returns
    0.0 (no trend can be estimated).
    """

    def compute(self, result: RetrievalResult) -> float:
        if result.count < 2:
            return 0.0

        times = np.array([m.timestamp for m in result.memories], dtype=np.float64)
        errors = np.array(
            [m.prediction_error for m in result.memories], dtype=np.float64
        )

        t = times - times.mean()
        denom = float(np.dot(t, t))
        if denom == 0.0:  # all timestamps identical => no temporal trend
            return 0.0

        slope = float(np.dot(t, errors - errors.mean()) / denom)
        return slope
