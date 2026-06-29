"""TabularQPolicy — the decider for POC 0 (TODO.md G.4).

Plain tabular Q-learning over the ``N×N`` grid states. Deliberately the simplest,
most *inspectable* policy possible: every state-action value can be printed and
eyeballed. It is a throwaway stand-in for the eventual DreamerV3 actor — chosen
for clarity while we validate the *drive*, not for code reuse. (We do NOT repeat
the hallucinator's "0% throwaway" claim; the world and this policy are scaffolding.)

The crucial part is *what the reward is*, and that is supplied by the caller, not
baked in here:

* feed it **learning progress** (TODO.md G.3) → the real agent that escapes the
  noise trap.
* feed it **raw prediction error** → the control agent that gets hypnotised.

Same policy code, two reward sources — that is exactly the A/B in TODO.md G.6.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .env import ACTIONS

Cell = Tuple[int, int]


class TabularQPolicy:
    """Epsilon-greedy tabular Q-learning over grid cells (TODO.md G.4).

    Args:
        size: Grid side length ``N`` (state space is ``N×N``).
        epsilon: Exploration rate for epsilon-greedy action selection.
        alpha: Q-learning step size.
        gamma: Discount factor.
        seed: RNG seed for exploration / tie-breaking.
    """

    def __init__(
        self,
        size: int,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.9,
        seed: int = 0,
    ) -> None:
        self.size = int(size)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self._rng = np.random.default_rng(seed)
        # Q[x, y, action].
        self.Q = np.zeros((self.size, self.size, len(ACTIONS)), dtype=np.float32)

    def choose(self, pos: Cell) -> int:
        """Epsilon-greedy action for ``pos`` (random tie-break among best)."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(len(ACTIONS)))
        x, y = pos
        q = self.Q[x, y]
        best = np.flatnonzero(q == q.max())
        return int(self._rng.choice(best))

    def update(
        self, pos: Cell, action: int, reward: float, next_pos: Cell
    ) -> None:
        """Standard Q-learning update toward ``reward + gamma * max_a' Q(next)``."""
        x, y = pos
        nx, ny = next_pos
        best_next = float(self.Q[nx, ny].max())
        target = reward + self.gamma * best_next
        self.Q[x, y, action] += self.alpha * (target - self.Q[x, y, action])
