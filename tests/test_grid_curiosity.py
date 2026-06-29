"""Tests for the ForwardModel curiosity engine (TODO.md G.2)."""

import numpy as np

from src.grid.curiosity import ForwardModel
from src.grid.env import OBS_DIM


def test_predict_shape():
    model = ForwardModel(seed=0)
    obs = np.array([0.1, 0.2, 0.0], dtype=np.float32)
    pred = model.predict(obs, action=0)
    assert pred.shape == (OBS_DIM,)


def test_error_drops_on_deterministic_transition():
    """Repeatedly learning one fixed transition drives its error toward ~0."""
    model = ForwardModel(seed=0, lr=0.1)
    obs = np.array([0.3, 0.4, 0.0], dtype=np.float32)
    next_obs = np.array([0.35, 0.4, 0.0], dtype=np.float32)

    first = model.prediction_error(obs, 3, next_obs)
    for _ in range(500):
        model.update(obs, 3, next_obs)
    last = model.prediction_error(obs, 3, next_obs)

    assert last < first
    assert last < 1e-3  # effectively learned


def test_error_stays_high_on_noise_target():
    """A random target every step can't be learned — error never collapses."""
    model = ForwardModel(seed=0, lr=0.1)
    rng = np.random.default_rng(7)
    obs = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    recent = []
    for _ in range(500):
        # Same (obs, action) but the noise channel of the target is random.
        next_obs = np.array([0.5, 0.5, rng.random()], dtype=np.float32)
        err = model.update(obs, 0, next_obs)
        recent.append(err)

    # Error over the last 100 steps stays clearly non-zero (unlearnable).
    assert np.mean(recent[-100:]) > 1e-2


def test_distinct_actions_can_predict_distinct_outcomes():
    """The action input matters: same obs, different action -> different target."""
    model = ForwardModel(seed=1, lr=0.1)
    obs = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    left_target = np.array([0.45, 0.5, 0.0], dtype=np.float32)
    right_target = np.array([0.55, 0.5, 0.0], dtype=np.float32)
    for _ in range(800):
        model.update(obs, 2, left_target)
        model.update(obs, 3, right_target)
    assert model.prediction_error(obs, 2, left_target) < 1e-2
    assert model.prediction_error(obs, 3, right_target) < 1e-2
