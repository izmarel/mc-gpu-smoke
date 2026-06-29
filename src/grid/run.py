"""Experiment runner + ASCII telemetry for POC 0 (TODO.md G.5, G.6, G.7).

Two experiments, both reported with terminal-printable heatmaps (no matplotlib):

* :func:`run_clean` (G.5) — clean grid, learning-progress agent. Proves pure
  curiosity explores the whole world, learns it (mean error falls), then has
  nothing left pulling it (it goes calm).
* :func:`run_noise_trap` (G.6) — noise cell ON, two agents on the same world:
  a CONTROL agent rewarded by raw surprise (should get stuck on the noise cell)
  and the REAL agent rewarded by learning progress (should sample it and leave).
  This is the project's lie-detector.

Run directly:  ``python -m src.grid.run``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .curiosity import TabularForwardModel
from .env import GridWorld
from .policy import TabularQPolicy
from .progress import LearningProgress

Cell = Tuple[int, int]

#: Reward modes for an agent run.
REWARD_LEARNING_PROGRESS = "learning_progress"
REWARD_RAW_SURPRISE = "raw_surprise"

#: A cell counts as "still learnable" if its learning progress exceeds this. Below
#: it (once enough samples exist) the cell is either mastered (low surprise) or
#: noise (high surprise but flat) — both cases the real agent should disengage from.
LEARNABLE_EPS = 0.01


@dataclass
class RunResult:
    """Outcome of one agent run (telemetry for G.5-G.7)."""

    size: int
    steps: int
    reward_mode: str
    noise_cell: Optional[Cell]
    visits: np.ndarray                      # (size, size) int visit counts
    coverage_curve: List[float] = field(default_factory=list)  # fraction seen over time
    mean_error_first: float = 0.0           # mean surprise over first 10% of steps
    mean_error_last: float = 0.0            # mean surprise over last 10% of steps
    noise_cell_visits: int = 0              # times the noise cell was entered

    @property
    def coverage(self) -> float:
        """Fraction of cells visited at least once."""
        return float((self.visits > 0).sum()) / (self.size * self.size)

    @property
    def noise_cell_share(self) -> float:
        """Share of all steps spent entering the noise cell (0 if no noise cell)."""
        total = int(self.visits.sum())
        return self.noise_cell_visits / total if total else 0.0


def run_agent(
    size: int = 20,
    steps: int = 20_000,
    reward_mode: str = REWARD_LEARNING_PROGRESS,
    noise_cell: Optional[Cell] = None,
    seed: int = 0,
) -> RunResult:
    """Run one agent for ``steps`` and collect telemetry.

    The loop is the whole POC in miniature: observe -> choose -> act -> measure
    surprise (and learn) -> turn surprise into the reward -> update the policy.
    """
    env = GridWorld(size=size, noise_cell=noise_cell, seed=seed)
    model = TabularForwardModel()
    progress = LearningProgress()
    policy = TabularQPolicy(size=size, seed=seed)

    visits = np.zeros((size, size), dtype=np.int64)
    seen = np.zeros((size, size), dtype=bool)
    coverage_curve: List[float] = []
    errors: List[float] = []
    noise_visits = 0

    obs = env.reset()
    pos = env.pos
    sample_every = max(1, steps // 200)

    for t in range(steps):
        action = policy.choose(pos)
        next_obs, info = env.step(action)
        next_pos = info["pos"]

        # Surprise + online learning (pre-update error is the surprise). Curiosity
        # is keyed on (current cell, action); surprise is attributed to the cell
        # actually entered, which is where it is "felt".
        error = model.update(pos, action, next_obs)
        progress.record(next_pos, error)
        errors.append(error)

        # The reward is the ONLY difference between the two agents (G.6).
        if reward_mode == REWARD_RAW_SURPRISE:
            # CONTROL: chase raw surprise. Novel AND noisy cells both look great
            # forever -> gets hypnotised by the noise cell.
            reward = error
        elif reward_mode == REWARD_LEARNING_PROGRESS:
            # REAL: chase surprise ONLY where learning is still happening. A cell
            # that's been sampled enough and has flat progress is either mastered
            # or noise -> disengage (reward 0). This is the concept's loop:
            # explore by novelty, stay while learning, leave when not.
            still_novel = progress.samples(next_pos) < progress.min_samples
            still_learning = progress.progress(next_pos) > LEARNABLE_EPS
            reward = error if (still_novel or still_learning) else 0.0
        else:
            raise ValueError(f"unknown reward_mode {reward_mode!r}")

        policy.update(pos, action, reward, next_pos)

        visits[next_pos] += 1
        seen[next_pos] = True
        if noise_cell is not None and next_pos == noise_cell:
            noise_visits += 1
        if t % sample_every == 0:
            coverage_curve.append(float(seen.sum()) / (size * size))

        obs, pos = next_obs, next_pos

    window = max(1, steps // 10)
    return RunResult(
        size=size,
        steps=steps,
        reward_mode=reward_mode,
        noise_cell=noise_cell,
        visits=visits,
        coverage_curve=coverage_curve,
        mean_error_first=float(np.mean(errors[:window])),
        mean_error_last=float(np.mean(errors[-window:])),
        noise_cell_visits=noise_visits,
    )


# --- telemetry (TODO.md G.7) --------------------------------------------------

_RAMP = " .:-=+*#%@"


def ascii_heatmap(grid: np.ndarray, mark: Optional[Cell] = None) -> str:
    """Render a 2-D array as an ASCII heatmap (low -> high == ' ' -> '@').

    ``mark`` (e.g. the noise cell) is drawn as ``N`` so it stands out.
    """
    g = np.asarray(grid, dtype=np.float64)
    hi = g.max()
    norm = g / hi if hi > 0 else g
    size = g.shape[0]
    lines = []
    for y in range(size):
        row = []
        for x in range(size):
            if mark is not None and (x, y) == mark:
                row.append("N")
                continue
            idx = int(norm[x, y] * (len(_RAMP) - 1))
            row.append(_RAMP[idx])
        lines.append("".join(row))
    return "\n".join(lines)


def run_clean(size: int = 20, steps: int = 20_000, seed: int = 0) -> RunResult:
    """Experiment 1 (G.5): clean grid, learning-progress agent."""
    res = run_agent(size=size, steps=steps, seed=seed)
    print(f"\n=== G.5 CLEAN GRID  ({size}x{size}, {steps} steps) ===")
    print(f"  coverage:        {res.coverage*100:.1f}%  ({int((res.visits>0).sum())}/{size*size} cells)")
    print(f"  mean surprise:   first 10% = {res.mean_error_first:.4f}  ->  last 10% = {res.mean_error_last:.4f}")
    drop = (1 - res.mean_error_last / res.mean_error_first) * 100 if res.mean_error_first else 0
    print(f"  surprise dropped {drop:.1f}%  (it learned the world)")
    print("  visit heatmap:")
    print(ascii_heatmap(res.visits))
    return res


def run_noise_trap(
    size: int = 20, steps: int = 20_000, noise_cell: Optional[Cell] = None, seed: int = 0
) -> Dict[str, RunResult]:
    """Experiment 2 (G.6): noise cell ON, control vs real agent — the lie detector."""
    if noise_cell is None:
        noise_cell = (size // 2, size // 2)
    control = run_agent(
        size=size, steps=steps, reward_mode=REWARD_RAW_SURPRISE,
        noise_cell=noise_cell, seed=seed,
    )
    real = run_agent(
        size=size, steps=steps, reward_mode=REWARD_LEARNING_PROGRESS,
        noise_cell=noise_cell, seed=seed,
    )
    fair_share = 1.0 / (size * size)
    print(f"\n=== G.6 NOISE TRAP  (noise cell at {noise_cell}, fair share = {fair_share*100:.2f}%) ===")
    print(f"  CONTROL (chases raw surprise):     {control.noise_cell_share*100:5.2f}% of steps on noise cell")
    print(f"  REAL    (chases learning progress):{real.noise_cell_share*100:5.2f}% of steps on noise cell")
    print(f"  control / real ratio: {control.noise_cell_share / real.noise_cell_share:.1f}x more time stuck" if real.noise_cell_share else "  real agent essentially never lingers")
    print("  CONTROL visit heatmap (N = noise cell):")
    print(ascii_heatmap(control.visits, mark=noise_cell))
    print("  REAL visit heatmap (N = noise cell):")
    print(ascii_heatmap(real.visits, mark=noise_cell))
    return {"control": control, "real": real}


if __name__ == "__main__":
    run_clean()
    run_noise_trap()
