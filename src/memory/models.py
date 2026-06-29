"""Data structures for the unified vector-graph memory (POC 1, TODO 1.2).

The memory is a single graph of nodes connected by typed edges:

    state --ACTION--> action
    state --RESULT--> next_state
    state --CONTEXT--> retrieved_state

Two node kinds live in the graph:

* :class:`MemoryNode` — a snapshot of experience. Carries the projected state
  vector that the vector index is built over, plus the prediction error,
  timestamp and optional internal-monologue text for that moment.
* :class:`ActionNode` — an action that was taken from a state. Carries the
  action vector and a coarse ``action_type`` label.

These types are backend-agnostic: they describe *what* is stored, not *how*.
The storage layer (:mod:`src.memory.storage`) is responsible for persisting
them, whether the backend is LadybugDB, FAISS+NetworkX, or the in-memory
reference implementation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np


def new_node_id() -> str:
    """Return a fresh unique node identifier."""
    return str(uuid.uuid4())


class EdgeType(str, Enum):
    """The three typed edges that wire experience together.

    ``str`` mixin so the enum serialises to its name transparently in JSON and
    compares equal to the raw string (``EdgeType.ACTION == "ACTION"``).
    """

    #: state -> action that was taken from it.
    ACTION = "ACTION"
    #: state -> the next state that followed (the consequence chain).
    RESULT = "RESULT"
    #: state -> a memory node that was retrieved/active when this happened.
    CONTEXT = "CONTEXT"


def _coerce_vector(vector: Any, *, name: str) -> np.ndarray:
    """Coerce ``vector`` to a contiguous float32 1-D array, or raise."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape!r}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return np.ascontiguousarray(arr)


@dataclass
class MemoryNode:
    """A single moment of experience stored in the graph.

    Attributes:
        state_vector: Projected state vector (e.g. 768 or 1792 dims depending
            on POC stage). This is what the vector index is built over.
        prediction_error: How surprising this moment was. Higher = more
            surprising; used for confidence, novelty and pruning decisions.
        timestamp: Unix time (seconds) of when this moment happened.
        text_metadata: Optional internal-monologue text active at this moment.
        node_id: Stable unique identifier (UUID4 string by default).
    """

    state_vector: np.ndarray
    prediction_error: float
    timestamp: float
    text_metadata: Optional[str] = None
    node_id: str = field(default_factory=new_node_id)

    def __post_init__(self) -> None:
        self.state_vector = _coerce_vector(self.state_vector, name="state_vector")
        self.prediction_error = float(self.prediction_error)
        self.timestamp = float(self.timestamp)

    @property
    def dim(self) -> int:
        """Dimensionality of the state vector."""
        return int(self.state_vector.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict (numpy array -> list)."""
        return {
            "kind": "MemoryNode",
            "node_id": self.node_id,
            "state_vector": self.state_vector.tolist(),
            "prediction_error": self.prediction_error,
            "timestamp": self.timestamp,
            "text_metadata": self.text_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            state_vector=np.asarray(data["state_vector"], dtype=np.float32),
            prediction_error=data["prediction_error"],
            timestamp=data["timestamp"],
            text_metadata=data.get("text_metadata"),
            node_id=data["node_id"],
        )


@dataclass
class ActionNode:
    """An action taken from a state, reachable via an ACTION edge.

    Attributes:
        action_vector: Vector encoding of the action (motor command, discretised
            token embedding, etc.).
        action_type: Coarse label for the action ("move", "click", "wait", ...).
        node_id: Stable unique identifier (UUID4 string by default).
    """

    action_vector: np.ndarray
    action_type: str
    node_id: str = field(default_factory=new_node_id)

    def __post_init__(self) -> None:
        self.action_vector = _coerce_vector(self.action_vector, name="action_vector")
        self.action_type = str(self.action_type)

    @property
    def dim(self) -> int:
        """Dimensionality of the action vector."""
        return int(self.action_vector.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict (numpy array -> list)."""
        return {
            "kind": "ActionNode",
            "node_id": self.node_id,
            "action_vector": self.action_vector.tolist(),
            "action_type": self.action_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionNode":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            action_vector=np.asarray(data["action_vector"], dtype=np.float32),
            action_type=data["action_type"],
            node_id=data["node_id"],
        )


#: Any node that can live in the graph.
Node = Union[MemoryNode, ActionNode]


@dataclass
class Edge:
    """A typed, directed edge between two nodes.

    Attributes:
        source_id: ``node_id`` of the source node.
        target_id: ``node_id`` of the target node.
        edge_type: One of :class:`EdgeType`.
        metadata: Optional free-form annotation (e.g. edge strength, when added).
    """

    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept raw strings and coerce to the enum for uniform handling.
        self.edge_type = EdgeType(self.edge_type)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            metadata=data.get("metadata", {}),
        )


def node_from_dict(data: Dict[str, Any]) -> Node:
    """Reconstruct a :class:`MemoryNode` or :class:`ActionNode` from a dict.

    Dispatches on the ``"kind"`` field written by ``to_dict``.
    """
    kind = data.get("kind")
    if kind == "MemoryNode":
        return MemoryNode.from_dict(data)
    if kind == "ActionNode":
        return ActionNode.from_dict(data)
    raise ValueError(f"unknown node kind: {kind!r}")
