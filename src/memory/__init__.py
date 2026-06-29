"""Unified vector-graph memory for the Artificial Consciousness MVP (POC 1).

Public API for the memory subsystem. Import data structures, the storage
interface + reference backend, and the retrieval/derived-signal layer from here:

    from src.memory import (
        MemoryNode, ActionNode, Edge, EdgeType,
        InMemoryStore, Retriever,
        SimpleConfidence, SimpleNovelty, LinearLearningProgress,
    )
"""

from .models import (
    ActionNode,
    Edge,
    EdgeType,
    MemoryNode,
    Node,
    new_node_id,
    node_from_dict,
)
from .retrieval import (
    ConfidenceComputer,
    LearningProgressComputer,
    LinearLearningProgress,
    NoveltyComputer,
    RetrievalResult,
    RetrievedMemory,
    Retriever,
    SimpleConfidence,
    SimpleNovelty,
)
from .faiss_networkx_store import FaissNetworkXStore
from .persistence import DEFAULT_INTERVAL, AutoPersister
from .prune import (
    PruneConfig,
    PruneStats,
    prune,
    run_sleep_cycle,
    should_sleep,
)
from .storage import (
    DistanceMetric,
    InMemoryStore,
    MemoryStore,
    SimilarityHit,
    cosine_distance,
    l2_distance,
)

__all__ = [
    # models
    "ActionNode",
    "Edge",
    "EdgeType",
    "MemoryNode",
    "Node",
    "new_node_id",
    "node_from_dict",
    # storage
    "AutoPersister",
    "DEFAULT_INTERVAL",
    "DistanceMetric",
    "FaissNetworkXStore",
    "InMemoryStore",
    "MemoryStore",
    "SimilarityHit",
    "cosine_distance",
    "l2_distance",
    # retrieval
    "ConfidenceComputer",
    "LearningProgressComputer",
    "LinearLearningProgress",
    "NoveltyComputer",
    "RetrievalResult",
    "RetrievedMemory",
    "Retriever",
    "SimpleConfidence",
    "SimpleNovelty",
    # prune / sleep
    "PruneConfig",
    "PruneStats",
    "prune",
    "run_sleep_cycle",
    "should_sleep",
]
