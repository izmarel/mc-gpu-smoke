"""LearningProgress — the actual reward signal for POC 0 (TODO.md G.3).

The single most important correction over the hallucinator's design: the drive
rewards **learning progress**, NOT raw surprise.

* Raw surprise (prediction error) at the noise cell stays high *forever* — a
  policy that chases it gets hypnotised (the "noisy-TV" trap).
* Learning progress asks a different question: *is my error for this kind of
  state going DOWN over time?* A normal cell: error falls as the model learns it
  → positive progress → worth visiting, briefly. The noise cell: error never
  falls → ~0 progress → not worth visiting. So the trap stops being a trap.

Implementation mirrors the concept in ``src/memory/retrieval.py``
(``LinearLearningProgress``, a slope of error vs time) but kept dead simple here:
per cell, keep a short rolling window of recent errors and report
``mean(older half) - mean(recent half)`` (positive = improving).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

Cell = Tuple[int, int]


class LearningProgress:
    """Per-cell "am I getting better at predicting this?" signal (TODO.md G.3).

    Args:
        window: How many recent errors to keep per cell. The window is split into
            an older half and a recent half to estimate the trend.
        min_samples: Minimum errors recorded for a cell before a non-zero progress
            is reported (need both halves populated to compare).
    """

    def __init__(self, window: int = 20, min_samples: int = 4) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = int(window)
        self.min_samples = max(2, int(min_samples))
        self._errors: Dict[Cell, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    def record(self, cell: Cell, error: float) -> None:
        """Append the latest prediction error observed for ``cell``."""
        self._errors[cell].append(float(error))

    def samples(self, cell: Cell) -> int:
        """How many errors have been recorded for ``cell`` (within the window)."""
        errs = self._errors.get(cell)
        return len(errs) if errs else 0

    def progress(self, cell: Cell) -> float:
        """Learning progress for ``cell``: positive == error trending down.

        Returns 0.0 until at least ``min_samples`` errors exist (no trend yet),
        which is also what the noise cell converges to (flat, high error).
        """
        errs = self._errors.get(cell)
        if errs is None or len(errs) < self.min_samples:
            return 0.0
        data = list(errs)
        mid = len(data) // 2
        older = data[:mid]
        recent = data[mid:]
        if not older or not recent:
            return 0.0
        return (sum(older) / len(older)) - (sum(recent) / len(recent))

    def mean_error(self, cell: Cell) -> float:
        """Mean recent error for ``cell`` (0.0 if unseen) — for telemetry."""
        errs = self._errors.get(cell)
        if not errs:
            return 0.0
        return sum(errs) / len(errs)
