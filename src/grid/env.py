"""GridWorld — the world for POC 0 (TODO.md G.1).

A bare N×N grid with NO reward and NO survival/termination. The only thing the
world does is move the agent and report what it observes. Everything that drives
behaviour (curiosity, learning progress) lives outside the world, in the agent.

Design choices that matter:

* **No reward field.** ``step`` returns ``(obs, info)``, never a reward. The drive
  is intrinsic (curiosity), so the environment must not leak any external goal.
* **No death / battery / food.** Rejected on purpose — survival pressure breeds
  adversarial behaviour and is not this project's drive (see PROJECT_CONTEXT).
* **Walls clamp** (the agent stays put when it walks into an edge), they do not
  wrap. Walking into a wall is itself a predictable, learnable transition.
* **The noise cell** is the deliberately-planted trap for TODO.md G.6. When the
  agent stands on it, the observation carries a fresh random value every step, so
  that cell is *unpredictable forever*. A naive "chase raw surprise" agent gets
  hypnotised by it; a "chase learning progress" agent learns it can't be learned
  and leaves. Default ``None`` (no trap) for the clean experiment (G.5).

Observation layout (3 floats): ``[x / (N-1), y / (N-1), noise_channel]`` where
``noise_channel`` is a fresh uniform draw in ``[0, 1)`` when the agent is on the
noise cell and ``0.0`` everywhere else.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

#: Discrete action set. Index -> (dx, dy). Order is fixed; the forward model and
#: policy one-hot/encode actions by this index.
ACTIONS: Tuple[Tuple[int, int], ...] = (
    (0, -1),  # 0 up
    (0, 1),   # 1 down
    (-1, 0),  # 2 left
    (1, 0),   # 3 right
)

#: Number of floats in an observation vector.
OBS_DIM = 3

Cell = Tuple[int, int]


class GridWorld:
    """An N×N grid with intrinsic-only dynamics (TODO.md G.1).

    Args:
        size: Grid side length ``N`` (the world is ``N×N``). Default 20.
        noise_cell: ``(x, y)`` of the unpredictable "noise" cell, or ``None`` for
            a clean world. On this cell the observation's noise channel is random.
        seed: RNG seed for the start position and the noise channel.
    """

    def __init__(
        self,
        size: int = 20,
        noise_cell: Optional[Cell] = None,
        seed: int = 0,
    ) -> None:
        if size < 2:
            raise ValueError("grid size must be >= 2")
        if noise_cell is not None and not self._in_bounds(noise_cell, size):
            raise ValueError(f"noise_cell {noise_cell!r} out of bounds for size {size}")
        self.size = int(size)
        self.noise_cell = noise_cell
        self._rng = np.random.default_rng(seed)
        self._pos: Cell = (0, 0)
        self.reset()

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _in_bounds(cell: Cell, size: int) -> bool:
        x, y = cell
        return 0 <= x < size and 0 <= y < size

    def _observe(self, pos: Cell) -> np.ndarray:
        """Build the observation vector for ``pos`` (random channel on noise cell)."""
        x, y = pos
        denom = self.size - 1
        noise = float(self._rng.random()) if pos == self.noise_cell else 0.0
        return np.array([x / denom, y / denom, noise], dtype=np.float32)

    # --- gym-style API --------------------------------------------------------

    @property
    def pos(self) -> Cell:
        """Current agent cell ``(x, y)``."""
        return self._pos

    def reset(self, pos: Optional[Cell] = None) -> np.ndarray:
        """Place the agent (random cell by default) and return the observation."""
        if pos is None:
            x = int(self._rng.integers(self.size))
            y = int(self._rng.integers(self.size))
            pos = (x, y)
        elif not self._in_bounds(pos, self.size):
            raise ValueError(f"reset pos {pos!r} out of bounds")
        self._pos = pos
        return self._observe(self._pos)

    def step(self, action: int) -> Tuple[np.ndarray, Dict]:
        """Apply ``action``; return ``(obs, info)``. No reward, ever.

        ``info`` carries ``"pos"`` (the resulting cell) so callers can attribute
        prediction error to the cell that was actually entered.
        """
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"invalid action {action!r}")
        dx, dy = ACTIONS[action]
        x = min(self.size - 1, max(0, self._pos[0] + dx))
        y = min(self.size - 1, max(0, self._pos[1] + dy))
        self._pos = (x, y)
        return self._observe(self._pos), {"pos": self._pos}
