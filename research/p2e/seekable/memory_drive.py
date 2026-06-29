"""Stage 1: drive the agent from the REAL memory layer (src/memory), not inline math.

Same seekable-noise world / neural forward model / tabular policy as experiment.py.
The ONLY change: the intrinsic reward now comes from the actual built memory stack —
FaissNetworkXStore + Retriever + LinearLearningProgress + SimpleNovelty. This is the
literal integration loop planned for DreamerV3, run cheaply on the Mac first:

    predict -> measure error -> STORE MemoryNode(state, error, t)
            -> RETRIEVE similar past states -> LinearLearningProgress slope = reward

What it shows:
  1. Does the memory graph reproduce the learning-progress drive? -> memory-driven agent
     should avoid the noise TV (~chance) like the inline 'progress' agent, while 'raw'
     stays trapped. If so, the built memory layer IS the drive engine, proven end-to-end.
  2. Per-step memory overhead (ms) -> the one real performance unknown for Stage 2.

NOTE on the state vector: memory keys on each cell's STABLE identity (its fixed texture),
the toy stand-in for the DreamerV3 latent. The prediction error still comes from the
actual (noise-corrupted on the TV) observation. That mirrors the real architecture:
a stable latent identity + a surprise signal that can be unpredictable. Stage 2 tests
whether the real RSSM latent provides that identity; Stage 1 tests the drive LOGIC.

Run:  PYTHONPATH=. python research/p2e/seekable/memory_drive.py   (from repo root)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# make both `src.memory` (repo root) and `experiment`/`collapse_check` (this dir) importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import experiment as E  # noqa: E402  world, view(), QPolicy, run_agent, NOISE_TILE, ...
from collapse_check import MLP, build_world, step  # noqa: E402
from src.memory import (  # noqa: E402
    FaissNetworkXStore,
    LinearLearningProgress,
    MemoryNode,
    Retriever,
    SimpleNovelty,
)

STEPS = 40_000   # match experiment.py: the raw-surprise trap needs ~40k to develop
                 # (memory overhead measured at ~0.26 ms/step, so the full horizon is cheap)


def run_memory_agent(seed=0):
    world = build_world(seed=0)
    model = MLP(seed=seed)
    pol = E.QPolicy(seed)
    rng = np.random.default_rng(1000 + seed)

    store = FaissNetworkXStore()
    retriever = Retriever(store)
    lp = LinearLearningProgress()
    nov = SimpleNovelty()

    noise_steps = 0
    mem_time = 0.0
    pos = (0, 0)
    v = E.view(world, pos, rng)

    for t in range(STEPS):
        a = pol.choose(pos)
        nxt = step(pos, a)
        nv = E.view(world, nxt, rng)
        err = model.update(v, a, nv)  # surprise from the actual (maybe noisy) observation

        # --- the real memory loop (timed) ---
        t0 = time.perf_counter()
        ident = world[nxt[0], nxt[1]]  # stable identity of the entered cell (latent stand-in)
        store.add_node(MemoryNode(state_vector=ident, prediction_error=err, timestamp=float(t)))
        result = retriever.retrieve(ident, k=8)
        if result.count < 4:
            reward = nov.compute(result)             # bootstrap: explore the genuinely new
        else:
            reward = max(0.0, -lp.compute(result))   # else: reward only where error is FALLING
        mem_time += time.perf_counter() - t0
        # ------------------------------------

        pol.update(pos, a, reward, nxt)
        if nxt == E.NOISE_TILE:
            noise_steps += 1
        pos, v = nxt, nv

    return {
        "noise_share": noise_steps / STEPS,
        "ms_per_step": mem_time / STEPS * 1000.0,
        "nodes": store.node_count(),
    }


def main():
    E.STEPS = STEPS  # run the reference agents at the same horizon for a fair comparison
    fair = 1.0 / (E.N * E.N)
    print(f"=== STAGE 1: memory-driven drive vs references ({E.N}x{E.N}, TV at {E.NOISE_TILE}, {STEPS} steps) ===")
    print(f"fair share on TV by chance = {fair*100:.1f}%\n")

    raw = E.run_agent("raw", seed=0)["noise_share"]
    inline = E.run_agent("progress", seed=0)["noise_share"]
    mem = run_memory_agent(seed=0)

    print(f"  raw surprise (reference) : {raw*100:5.1f}% on TV")
    print(f"  inline progress (ref)    : {inline*100:5.1f}% on TV")
    print(f"  MEMORY-DRIVEN (src/memory): {mem['noise_share']*100:5.1f}% on TV")
    print(f"     stored {mem['nodes']} nodes | {mem['ms_per_step']:.2f} ms/step memory overhead\n")

    if raw > 3 * fair and mem["noise_share"] < raw / 2:
        print("  RESULT: the memory graph reproduces the learning-progress drive.")
        print("          memory-driven agent AVOIDS the noise trap that raw surprise falls into")
        print("          -> the built src/memory layer IS the drive engine, proven end-to-end.")
    else:
        print("  RESULT: memory-driven agent did NOT reproduce the drive (trapped or no signal).")
        print("          Bug in the memory loop - fix here (free) before Stage 2.")


if __name__ == "__main__":
    main()
