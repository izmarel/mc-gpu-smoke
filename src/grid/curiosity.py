"""Curiosity engines for POC 0 (TODO.md G.2).

The curiosity engine predicts the *next* observation; its prediction error (MSE)
IS the surprise the whole drive is built on:

    surprise = MSE(predicted_next_obs, actual_next_obs)

Two implementations live here, and which one you use matters a LOT:

* :class:`TabularForwardModel` — **the one POC 0 uses.** Learns a separate
  prediction per (cell, action). An unvisited cell is maximally surprising (so
  curiosity genuinely drives exploration), a mastered cell collapses to ~0, and
  the noise cell stays surprising forever (random target). This is the setting in
  which "surprise" is a meaningful per-place signal — required for the G.6 trap.

* :class:`ForwardModel` — a tiny numpy MLP (kept for reference, NOT used by the
  experiments). **Empirical finding (2026-06-28):** on this grid the MLP learns
  the globally-linear dynamics almost instantly and prediction error collapses
  *everywhere at once* — surprise starts at ~0.001 and there is essentially no
  curiosity gradient left to explore with. A smooth function approximator
  generalises the surprise away. This is a real, known phenomenon (it is why
  curiosity methods like Plan2Explore use ensemble *disagreement*, not raw
  single-model error) and is exactly the harder problem the future neural
  world-model stage (POC 2.5+) has to solve. For POC 0 we isolate the *drive*
  mechanism with the tabular model and leave that harder problem to its own stage.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .env import ACTIONS, OBS_DIM

Cell = Tuple[int, int]


class TabularForwardModel:
    """Per-(cell, action) curiosity model — the POC 0 default (TODO.md G.2).

    Keeps a running mean of the observed next observation for each (cell, action)
    and reports how far the latest outcome was from that mean. A never-seen
    (cell, action) returns ``unseen_error`` (maximal surprise) so novelty pulls
    the agent to unexplored ground.

    Args:
        unseen_error: Surprise reported for a (cell, action) never seen before.
    """

    def __init__(self, unseen_error: float = 1.0) -> None:
        self.unseen_error = float(unseen_error)
        self._est: Dict[Tuple[Cell, int], np.ndarray] = {}
        self._count: Dict[Tuple[Cell, int], int] = {}

    def predict(self, cell: Cell, action: int) -> Optional[np.ndarray]:
        """Mean observed next obs for (cell, action), or ``None`` if never seen."""
        return self._est.get((cell, action))

    def prediction_error(self, cell: Cell, action: int, next_obs: np.ndarray) -> float:
        """Surprise for this transition (``unseen_error`` if never seen)."""
        est = self._est.get((cell, action))
        if est is None:
            return self.unseen_error
        diff = est - np.asarray(next_obs, dtype=np.float32)
        return float(np.mean(diff * diff))

    def update(self, cell: Cell, action: int, next_obs: np.ndarray) -> float:
        """Fold ``next_obs`` into the running mean; return the pre-update surprise.

        Deterministic cell: mean snaps to the truth in a couple of visits, so
        surprise falls 1.0 -> ~0 fast (learnable). Noise cell: the mean settles
        near the noise's expected value but each actual draw is random, so the
        per-step surprise stays at the noise variance forever (unlearnable).
        """
        key = (cell, action)
        target = np.asarray(next_obs, dtype=np.float32)
        est = self._est.get(key)
        if est is None:
            self._est[key] = target.copy()
            self._count[key] = 1
            return self.unseen_error
        diff = est - target
        error = float(np.mean(diff * diff))
        count = self._count[key] + 1
        self._count[key] = count
        est += (target - est) / count  # incremental mean
        return error


def _one_hot_action(action: int) -> np.ndarray:
    v = np.zeros(len(ACTIONS), dtype=np.float32)
    v[action] = 1.0
    return v


class ForwardModel:
    """One-hidden-layer tanh MLP: ``[obs ++ one_hot(action)] -> next_obs``.

    Args:
        hidden: Hidden layer width.
        lr: SGD learning rate for :meth:`update`.
        seed: RNG seed for weight initialisation.
    """

    def __init__(self, hidden: int = 64, lr: float = 0.05, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        in_dim = OBS_DIM + len(ACTIONS)
        out_dim = OBS_DIM
        # Small random init (scaled) for stable tanh activations.
        self.W1 = rng.standard_normal((in_dim, hidden)).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden, out_dim)).astype(np.float32) * 0.1
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.lr = float(lr)

    # --- inference ------------------------------------------------------------

    def _features(self, obs: np.ndarray, action: int) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs, dtype=np.float32), _one_hot_action(action)]
        )

    def _forward(self, x: np.ndarray):
        """Return (prediction, hidden_activation) for input row ``x`` (1-D)."""
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        out = h @ self.W2 + self.b2
        return out, h

    def predict(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Predicted next observation for ``(obs, action)``."""
        out, _ = self._forward(self._features(obs, action))
        return out

    def prediction_error(
        self, obs: np.ndarray, action: int, next_obs: np.ndarray
    ) -> float:
        """MSE between the predicted and actual next observation (the surprise)."""
        pred = self.predict(obs, action)
        diff = pred - np.asarray(next_obs, dtype=np.float32)
        return float(np.mean(diff * diff))

    # --- learning -------------------------------------------------------------

    def update(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> float:
        """One SGD step toward predicting ``next_obs``; return the pre-update MSE.

        The returned error is measured *before* the weight update, so it reflects
        how surprising the transition was given everything learned so far.
        """
        x = self._features(obs, action)
        y = np.asarray(next_obs, dtype=np.float32)

        out, h = self._forward(x)
        diff = out - y
        error = float(np.mean(diff * diff))

        # Backprop (MSE averaged over OBS_DIM outputs).
        dout = (2.0 / OBS_DIM) * diff            # (out_dim,)
        dW2 = np.outer(h, dout)                  # (hidden, out_dim)
        db2 = dout
        dh = dout @ self.W2.T                     # (hidden,)
        dz1 = dh * (1.0 - h * h)                  # tanh'
        dW1 = np.outer(x, dz1)                    # (in_dim, hidden)
        db1 = dz1

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return error
