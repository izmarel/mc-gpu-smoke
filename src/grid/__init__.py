"""POC 0 — Curiosity Grid (TODO.md "POC 0").

The simplest possible proof of the project's single drive: an agent driven purely
by *curiosity* (building a better model of its world), with NO survival pressure.

Pieces (see TODO.md G.1-G.7):

* :class:`GridWorld` — an N×N world with no reward and no death (``env``).
* :class:`ForwardModel` — predicts the next observation; its error is "surprise"
  (``curiosity``).
* :class:`LearningProgress` — turns the error history into "am I getting better?"
  — the actual reward signal, NOT raw surprise (``progress``).
* :class:`TabularQPolicy` — picks actions to maximise learning progress
  (``policy``).

The two experiments that validate the core bet live in :mod:`src.grid.run`.
"""

from .env import ACTIONS, GridWorld
from .curiosity import ForwardModel, TabularForwardModel
from .progress import LearningProgress
from .policy import TabularQPolicy

__all__ = [
    "ACTIONS",
    "GridWorld",
    "ForwardModel",
    "TabularForwardModel",
    "LearningProgress",
    "TabularQPolicy",
]
