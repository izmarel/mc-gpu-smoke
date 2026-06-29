"""Learning-progress drive — the intrinsic reward, computed from memory.

The reward is the same signal as src/memory's LinearLearningProgress + SimpleNovelty,
but computed in a BATCHED, vectorised way so the whole training batch (B*T latents) is
scored in one shot instead of a Python loop of single lookups.

Per gradient step (`batch_reward`):
  1. one faiss batch search of all latents against the stored past states (cosine);
  2. gather each query's neighbours' (prediction_error, timestamp) via numpy fancy-index;
  3. vectorised OLS slope of error-vs-time = learning progress; reward = max(0, -slope)
     where enough history exists, else SimpleNovelty as a cold-start bootstrap;
  4. one batch faiss add of the new latents (FIFO-capped).

On a TV/noise state errors stay high+flat -> slope ~0 -> reward ~0. On a learnable
state errors fall over time -> negative slope -> positive reward. Identical math to the
src/memory implementation, just vectorised.

NOTE: the drive only needs vector retrieval + each node's (error, timestamp); it never
reads the NetworkX graph (ACTION/RESULT edges) — those are used by the memory-injection
side (research/p2e/retrieval/inject.py), which is untouched. So the drive store is lean:
a faiss FLAT index + two numpy arrays. faiss is multi-threaded (shared-OpenMP env; see TODO.md).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import faiss


class MemoryDrive:
    """Vectorised learning-progress reward over a lean faiss-backed memory.

    Args:
        k: neighbours retrieved per query.
        min_history: min neighbours before learning-progress is trusted; below this,
            SimpleNovelty bootstraps exploration.
        max_nodes: FIFO cap on stored nodes (recent window; env MEM_DRIVE_MAX overrides).
        dim: latent dimensionality (set on first add if None).
        hnsw_m, ef_search: faiss HNSW build/search params (match src/memory defaults).
    """

    def __init__(self, k: int = 8, min_history: int = 4, max_nodes: Optional[int] = None,
                 dim: Optional[int] = None, hnsw_m: int = 32, ef_search: int = 64,
                 novelty_distance_scale: float = 0.5, novelty_expected_count: int = 5) -> None:
        self.k = int(k)
        self.min_history = int(min_history)
        self.max_nodes = (
            int(max_nodes) if max_nodes is not None
            else (int(os.environ["MEM_DRIVE_MAX"]) if os.environ.get("MEM_DRIVE_MAX") else None)
        )
        self.hnsw_m = int(hnsw_m)
        self.ef_search = int(ef_search)
        self._dscale = float(novelty_distance_scale)
        self._ecount = max(1, int(novelty_expected_count))
        self._dim = dim
        self._index: Optional[faiss.Index] = None
        # vectors live INSIDE faiss (reconstructed only on evict/persist — never copied
        # per step). Only the small (error, time) scalars are kept alongside.
        self._err = np.empty((0,), dtype=np.float32)
        self._time = np.empty((0,), dtype=np.float32)
        self._t = 0.0  # auto step counter when timestamp not supplied
        self.log: list = []

    # --- index management -----------------------------------------------------

    def _new_index(self, dim: int) -> faiss.Index:
        # FLAT (exact) cosine index: insert is instant (HNSW insert of 1024 vecs/step was
        # the measured bottleneck, ~5s); search is O(store) but the FIFO cap bounds it.
        return faiss.IndexFlatIP(dim)

    @property
    def _n(self) -> int:
        return self._err.shape[0]

    def node_count(self) -> int:
        return self._n

    def _all_vecs(self) -> np.ndarray:
        """Reconstruct stored vectors from faiss (only for evict/persist, not per step)."""
        if self._index is None or self._index.ntotal == 0:
            return np.empty((0, self._dim or 1), dtype=np.float32)
        return self._index.reconstruct_n(0, self._index.ntotal)

    def _rebuild_from(self, vecs: np.ndarray) -> None:
        self._index = self._new_index(self._dim)
        if len(vecs):
            self._index.add(np.ascontiguousarray(vecs, dtype=np.float32))

    def _add_batch(self, vecs_norm: np.ndarray, errors: np.ndarray, times: np.ndarray) -> None:
        if self._index is None:
            self._index = self._new_index(self._dim)
        self._index.add(np.ascontiguousarray(vecs_norm))      # O(batch) — faiss owns the vectors
        self._err = np.concatenate([self._err, errors.astype(np.float32)])    # tiny scalar arrays
        self._time = np.concatenate([self._time, times.astype(np.float32)])
        # FIFO cap: drop oldest, rebuild once when over cap+slack (reconstruct, not a kept copy)
        if self.max_nodes:
            slack = max(1, self.max_nodes // 10)
            if self._n > self.max_nodes + slack:
                drop = self._n - self.max_nodes
                kept = self._all_vecs()[drop:]
                self._err = self._err[drop:]
                self._time = self._time[drop:]
                self._rebuild_from(kept)

    # --- the reward -----------------------------------------------------------

    def batch_reward(self, latents: np.ndarray, errors: np.ndarray,
                     step: Optional[float] = None) -> np.ndarray:
        """Score a whole batch of latents in one shot. Returns rewards (N,) float32.

        All nodes added this call share timestamp `step` (the gradient-step index) — what
        the slope needs is error-over-time across DIFFERENT steps, so one timestamp per
        batch is correct (and cleaner than per-node).
        """
        L = np.ascontiguousarray(np.asarray(latents, dtype=np.float32))
        if L.ndim == 1:
            L = L[None]
        N = L.shape[0]
        if self._dim is None:
            self._dim = L.shape[1]
        Ln = L.copy()
        faiss.normalize_L2(Ln)
        errors = np.asarray(errors, dtype=np.float32).reshape(N)
        step = self._t if step is None else float(step)

        rewards = np.zeros(N, dtype=np.float32)
        n = self._n
        if n > 0:
            kq = min(self.k, n)
            sim, I = self._index.search(Ln, kq)          # sim:(N,kq) cosine, I positions or -1
            valid = I >= 0
            counts = valid.sum(1)
            Ic = np.where(valid, I, 0)
            nerr = self._err[Ic].astype(np.float64)
            ntime = self._time[Ic].astype(np.float64)
            m = valid.astype(np.float64)
            cnt = np.maximum(m.sum(1), 1.0)
            # vectorised OLS slope of error vs time (mean-centred), per row
            tmean = (ntime * m).sum(1) / cnt
            emean = (nerr * m).sum(1) / cnt
            tc = (ntime - tmean[:, None]) * m
            ec = (nerr - emean[:, None]) * m
            denom = (tc * tc).sum(1)
            num = (tc * ec).sum(1)
            slope = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
            lp_reward = np.maximum(0.0, -slope)
            # vectorised SimpleNovelty (cosine distance = 1 - sim; matches "0 at identical")
            dist = 1.0 - sim
            nearest = np.where(valid, dist, np.inf).min(1)
            nearest = np.where(np.isfinite(nearest), nearest, 1.0)
            dist_nov = 1.0 - np.exp(-nearest / self._dscale)
            sparsity = 1.0 - np.minimum(1.0, counts / self._ecount)
            novelty = np.clip(0.7 * dist_nov + 0.3 * sparsity, 0.0, 1.0)
            use_lp = counts >= self.min_history
            rewards = np.where(use_lp, lp_reward, novelty).astype(np.float32)
        else:
            rewards[:] = 1.0  # cold start: maximal novelty

        self._add_batch(Ln, errors, np.full(N, step, dtype=np.float32))
        self._t = step + 1.0
        self.log.append({"step": step, "n": self._n, "reward_mean": float(rewards.mean())})
        return rewards

    def reward(self, state_vector: np.ndarray, prediction_error: float,
               timestamp: float, action_vector=None, action_type: str = "act") -> Tuple[float, str]:
        """Single-state compatibility wrapper (used by tests / the legacy path)."""
        r = self.batch_reward(np.asarray(state_vector)[None], np.asarray([prediction_error]), timestamp)
        return float(r[0]), ""

    # --- persistence (resume / transfer) --------------------------------------

    def persist(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = f"{path}.tmp"
        np.savez(tmp, vecs=self._all_vecs(), err=self._err, time=self._time,
                 dim=np.array([self._dim if self._dim else 0]), t=np.array([self._t]))
        os.replace(tmp + ".npz", path)

    def load_store(self, path: str) -> None:
        d = np.load(path, allow_pickle=False)
        vecs = d["vecs"].astype(np.float32)
        self._err = d["err"].astype(np.float32)
        self._time = d["time"].astype(np.float32)
        self._dim = int(d["dim"][0]) or (vecs.shape[1] if vecs.size else None)
        self._t = float(d["t"][0])
        self._rebuild_from(vecs)

    def stats(self) -> Dict:
        return {"nodes": self._n, "t": self._t, "decisions_logged": len(self.log)}
