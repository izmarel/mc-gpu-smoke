"""Auto-persistence for the memory store (POC 1, TODO 1.10).

NetworkX + FAISS have no durability story of their own — nothing hits disk
unless we put it there. The main tick loop runs at ~30 fps, so an unbounded
in-RAM graph is one crash away from losing everything since startup. This module
bounds that loss to a fixed window.

:class:`AutoPersister` wraps any :class:`~src.memory.storage.MemoryStore` and a
snapshot path. The loop calls :meth:`AutoPersister.tick` once per tick; every
``interval`` ticks it writes an atomic snapshot (temp file + rename, inherited
from the store's :meth:`~src.memory.storage.MemoryStore.persist`). On startup
:meth:`AutoPersister.open` reloads the last snapshot if one exists.

Worst-case data loss on a hard crash is therefore at most ``interval`` ticks of
experience (default 100 ticks ≈ 3 s at 30 fps).
"""

from __future__ import annotations

import os
from typing import Callable

from .faiss_networkx_store import FaissNetworkXStore
from .storage import MemoryStore

#: Default persist cadence. 100 ticks ≈ 3 s at 30 fps (TODO 1.10).
DEFAULT_INTERVAL = 100


class AutoPersister:
    """Persist a :class:`MemoryStore` to disk every ``interval`` ticks.

    Args:
        store: The store to snapshot.
        path: Snapshot file path. Written atomically by the store.
        interval: Ticks between automatic snapshots (>= 1).

    Attributes:
        store: The wrapped store (mutate it directly: ``ap.store.add_node(...)``).
        path: Snapshot path.
        interval: Persist cadence in ticks.
    """

    def __init__(
        self,
        store: MemoryStore,
        path: str,
        interval: int = DEFAULT_INTERVAL,
    ) -> None:
        self.store = store
        self.path = path
        self.interval = max(1, int(interval))
        self._tick_count = 0
        self._ticks_since_persist = 0

    @classmethod
    def open(
        cls,
        path: str,
        store_factory: Callable[[], MemoryStore] = FaissNetworkXStore,
        interval: int = DEFAULT_INTERVAL,
    ) -> "AutoPersister":
        """Construct a persister, auto-loading ``path`` if a snapshot exists.

        This is the startup entry point: a fresh process picks up exactly where
        the last snapshot left it (TODO 1.10 "auto-load last persisted state").
        """
        store = store_factory()
        if os.path.exists(path):
            store.load(path)
        return cls(store, path, interval)

    def tick(self, n: int = 1) -> bool:
        """Advance the tick counter by ``n``; snapshot if the window elapsed.

        Returns ``True`` if a snapshot was written on this call. Call once per
        main-loop tick after the tick's memories have been added to the store.
        """
        self._tick_count += n
        self._ticks_since_persist += n
        if self._ticks_since_persist >= self.interval:
            self.persist_now()
            return True
        return False

    def persist_now(self) -> None:
        """Force an immediate atomic snapshot and reset the window.

        Use at a clean shutdown (or a sleep/prune boundary) to flush the tail of
        experience that hasn't hit a tick boundary yet.
        """
        self.store.persist(self.path)
        self._ticks_since_persist = 0

    @property
    def tick_count(self) -> int:
        """Total ticks observed since this persister was created."""
        return self._tick_count

    @property
    def ticks_since_persist(self) -> int:
        """Ticks since the last snapshot — the current crash-loss exposure."""
        return self._ticks_since_persist
