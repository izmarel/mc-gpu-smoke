"""Auto-persist tests (TODO 1.10).

Covers the tick-cadence logic, atomic snapshotting, and — the real requirement —
that memories survive a hard process kill: a child process writes nodes with
auto-persist enabled and is terminated without a clean shutdown, then a fresh
process reloads the snapshot and finds everything up to the last tick boundary.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from src.memory import AutoPersister, FaissNetworkXStore, MemoryNode

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mem(i: int) -> MemoryNode:
    vec = np.zeros(8, dtype=np.float32)
    vec[i % 8] = 1.0
    return MemoryNode(state_vector=vec, prediction_error=0.0, timestamp=float(i))


# --------------------------------------------------------------------------- #
# tick cadence
# --------------------------------------------------------------------------- #


def test_tick_persists_every_interval(tmp_path):
    path = str(tmp_path / "snap.pkl")
    ap = AutoPersister(FaissNetworkXStore(), path, interval=10)
    persisted_on = [t for t in range(1, 31) if (ap.store.add_node(_mem(t)), ap.tick())[1]]
    # Snapshot fires exactly at ticks 10, 20, 30.
    assert persisted_on == [10, 20, 30]
    assert ap.tick_count == 30
    assert ap.ticks_since_persist == 0


def test_ticks_since_persist_tracks_exposure(tmp_path):
    ap = AutoPersister(FaissNetworkXStore(), str(tmp_path / "s.pkl"), interval=100)
    for _ in range(37):
        ap.tick()
    assert ap.ticks_since_persist == 37  # crash here loses 37 ticks (< interval)


def test_persist_now_flushes_tail(tmp_path):
    path = str(tmp_path / "s.pkl")
    ap = AutoPersister(FaissNetworkXStore(), path, interval=100)
    for i in range(5):  # fewer than one interval — no auto snapshot yet
        ap.store.add_node(_mem(i))
        ap.tick()
    assert not Path(path).exists()
    ap.persist_now()
    assert Path(path).exists()
    reloaded = AutoPersister.open(path)
    assert reloaded.store.node_count() == 5


def test_open_loads_existing_snapshot(tmp_path):
    path = str(tmp_path / "s.pkl")
    ap = AutoPersister(FaissNetworkXStore(), path, interval=1)
    ap.store.add_node(_mem(0))
    ap.tick()  # writes snapshot
    fresh = AutoPersister.open(path, interval=50)
    assert fresh.store.node_count() == 1
    assert fresh.interval == 50


def test_open_missing_snapshot_starts_empty(tmp_path):
    ap = AutoPersister.open(str(tmp_path / "nope.pkl"))
    assert ap.store.node_count() == 0


# --------------------------------------------------------------------------- #
# survives a hard process kill (TODO 1.10 acceptance test)
# --------------------------------------------------------------------------- #


def test_survives_process_kill(tmp_path):
    """Write 1050 nodes with interval=100 in a child process, kill it, reload.

    The last auto-snapshot lands at tick 1000, so a fresh process must see 1000
    nodes — the 50 ticks since are the (bounded) crash loss.
    """
    path = tmp_path / "snap.pkl"
    writer = textwrap.dedent(
        f"""
        import os, numpy as np
        from src.memory import AutoPersister, FaissNetworkXStore, MemoryNode
        ap = AutoPersister(FaissNetworkXStore(), {str(path)!r}, interval=100)
        for i in range(1050):
            v = np.zeros(8, dtype=np.float32); v[i % 8] = 1.0
            ap.store.add_node(MemoryNode(state_vector=v, prediction_error=0.0, timestamp=float(i)))
            ap.tick()
        # Simulate a crash: exit hard without a clean persist_now().
        os._exit(0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", writer],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert path.exists(), "child never reached a persist boundary"

    # Fresh process: reload and verify the last persisted window survived.
    reloaded = AutoPersister.open(str(path))
    assert reloaded.store.node_count() == 1000
