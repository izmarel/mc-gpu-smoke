"""Seekable-noise experiment: raw surprise vs ensemble vs learning-progress (neural).

Built on the world validated by collapse_check.py (12x novelty signal, no collapse).
Adds the SEEKABLE noise trap and runs the three curiosity signals head-to-head.

World: 6x6 grid, each cell a fixed high-contrast binary "view". ONE cell is the
"noise TV": when the agent stands on it, its view is fresh random static EVERY
step (unpredictable forever). The agent can choose to walk onto it (and watch
static) or explore the rest. Same world for every agent; only the curiosity
signal differs.

Three signals (the intrinsic reward driving a tabular Q policy):
  * raw surprise        = one model's prediction error   -> expected to get TRAPPED.
  * ensemble disagreement= variance across K models       -> Plan2Explore's method.
  * learning progress    = is per-cell error trending DOWN -> YOUR method.
Plus a random-walk baseline (time on the TV by pure chance).

Metric: fraction of steps each agent spends ON the noise tile (the trap), and how
much of the rest of the world it explores. The test is VALID only if raw surprise
gets trapped (the canary); if it doesn't, the trap is too weak and must be hardened.

Run:  python research/p2e/seekable/experiment.py
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from collapse_check import ACTIONS, MLP, N, TEX, build_world, step

K = 5                       # ensemble size
TRAIN_WARMUP_LR = 0.03
STEPS = 40_000
NOISE_TILE = (3, 3)         # the seekable "TV"
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9


def view(world, pos, rng):
    """Observation at pos: fixed texture, or fresh random static on the noise tile."""
    if pos == NOISE_TILE:
        return (rng.random(TEX) < 0.5).astype(np.float32)
    return world[pos[0], pos[1]]


class Ensemble:
    """K independent MLP forward models trained on the same transitions."""

    def __init__(self, seed):
        self.models = [MLP(seed=seed * 100 + i) for i in range(K)]

    def update(self, v, a, target):
        for m in self.models:
            m.update(v, a, target)

    def predictions(self, v, a):
        return np.stack([m.predict(v, a) for m in self.models])  # (K, TEX)

    def raw_error(self, v, a, target):
        return self.models[0].error(v, a, target)

    def disagreement(self, v, a):
        # variance across the K models' predictions, averaged over output dims
        return float(self.predictions(v, a).var(axis=0).mean())


class QPolicy:
    def __init__(self, seed):
        self.Q = np.zeros((N, N, len(ACTIONS)), np.float32)
        self.rng = np.random.default_rng(seed)

    def choose(self, pos):
        if self.rng.random() < EPSILON:
            return int(self.rng.integers(len(ACTIONS)))
        q = self.Q[pos[0], pos[1]]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, pos, a, reward, nxt):
        best = float(self.Q[nxt[0], nxt[1]].max())
        self.Q[pos[0], pos[1], a] += ALPHA * (reward + GAMMA * best - self.Q[pos[0], pos[1], a])


def run_agent(mode, seed=0):
    """Run one agent. mode in {raw, ensemble, progress, random}. Returns telemetry."""
    world = build_world(seed=0)
    ens = Ensemble(seed)
    pol = QPolicy(seed)
    rng = np.random.default_rng(1000 + seed)
    err_hist = defaultdict(lambda: deque(maxlen=20))  # per-cell error history (for progress)

    visits = np.zeros((N, N), np.int64)
    noise_steps = 0
    pos = (0, 0)
    v = view(world, pos, rng)

    for t in range(STEPS):
        if mode == "random":
            a = int(rng.integers(len(ACTIONS)))
        else:
            a = pol.choose(pos)
        nxt = step(pos, a)
        nv = view(world, nxt, rng)

        pre_err = ens.raw_error(v, a, nv)  # surprise measured before learning
        ens.update(v, a, nv)
        err_hist[nxt].append(pre_err)

        if mode == "raw":
            reward = pre_err
        elif mode == "ensemble":
            reward = ens.disagreement(v, a)
        elif mode == "progress":
            h = err_hist[nxt]
            if len(h) >= 4:
                half = len(h) // 2
                reward = float(np.mean(list(h)[:half]) - np.mean(list(h)[half:]))
            else:
                reward = pre_err  # treat brand-new as worth a look until we can judge
        else:  # random
            reward = 0.0

        if mode != "random":
            pol.update(pos, a, reward, nxt)

        visits[nxt] += 1
        if nxt == NOISE_TILE:
            noise_steps += 1
        pos, v = nxt, nv

    return {
        "mode": mode,
        "noise_share": noise_steps / STEPS,
        "coverage": float((visits > 0).sum()) / (N * N),
        "visits": visits,
    }


def main():
    fair = 1.0 / (N * N)
    print(f"=== SEEKABLE-NOISE EXPERIMENT (6x6, TV at {NOISE_TILE}, {STEPS} steps) ===")
    print(f"fair share on TV by chance = {fair*100:.1f}%\n")
    results = {}
    for mode in ["random", "raw", "ensemble", "progress"]:
        r = run_agent(mode, seed=0)
        results[mode] = r
        print(f"  {mode:9}: {r['noise_share']*100:5.1f}% of steps on the TV | coverage {r['coverage']*100:4.0f}%")
    print()
    raw = results["raw"]["noise_share"]
    print("VALIDITY (canary): raw surprise must get trapped on the TV.")
    if raw > 3 * fair:
        print(f"  OK - raw surprise is stuck ({raw*100:.1f}% vs {fair*100:.1f}% chance). Trap is real.")
        for m in ["ensemble", "progress"]:
            s = results[m]["noise_share"]
            verdict = "AVOIDS the trap" if s < raw / 2 else "ALSO trapped"
            print(f"  {m}: {s*100:.1f}% on TV -> {verdict}")
    else:
        print(f"  WEAK: raw surprise not trapped ({raw*100:.1f}%). Trap too soft - harden it before trusting results.")


if __name__ == "__main__":
    main()
