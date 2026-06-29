"""Tests for GridWorld (TODO.md G.1)."""

import numpy as np
import pytest

from src.grid.env import ACTIONS, OBS_DIM, GridWorld


def test_reset_returns_valid_obs():
    env = GridWorld(size=10, seed=1)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    # Normalised position in [0, 1]; no noise cell so noise channel is 0.
    assert 0.0 <= obs[0] <= 1.0 and 0.0 <= obs[1] <= 1.0
    assert obs[2] == 0.0


def test_reset_to_explicit_cell():
    env = GridWorld(size=10, seed=1)
    obs = env.reset(pos=(0, 0))
    assert env.pos == (0, 0)
    assert obs[0] == 0.0 and obs[1] == 0.0


@pytest.mark.parametrize("action,delta", list(enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)])))
def test_actions_move_correctly(action, delta):
    env = GridWorld(size=10, seed=1)
    env.reset(pos=(5, 5))
    env.step(action)
    assert env.pos == (5 + delta[0], 5 + delta[1])


def test_walls_clamp_not_wrap():
    env = GridWorld(size=5, seed=1)
    env.reset(pos=(0, 0))
    env.step(0)  # up at the top edge
    assert env.pos == (0, 0)  # clamped, not wrapped to (0, 4)
    env.step(2)  # left at the left edge
    assert env.pos == (0, 0)
    env.reset(pos=(4, 4))
    env.step(1)  # down at the bottom edge
    env.step(3)  # right at the right edge
    assert env.pos == (4, 4)


def test_no_reward_field_in_step():
    env = GridWorld(size=5, seed=1)
    env.reset()
    out = env.step(1)
    # step returns exactly (obs, info) — never a reward (intrinsic drive only).
    assert len(out) == 2
    obs, info = out
    assert "pos" in info and info["pos"] == env.pos


def test_noise_channel_only_on_noise_cell_and_varies():
    noise = (2, 2)
    env = GridWorld(size=5, noise_cell=noise, seed=3)
    # Off the noise cell: channel is 0.
    off = env.reset(pos=(0, 0))
    assert off[2] == 0.0
    # On the noise cell: channel is non-zero and changes across observations.
    vals = set()
    for _ in range(20):
        obs = env.reset(pos=noise)
        vals.add(float(obs[2]))
    assert all(v != 0.0 for v in vals)
    assert len(vals) > 1  # genuinely random, not a constant


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        GridWorld(size=1)
    with pytest.raises(ValueError):
        GridWorld(size=5, noise_cell=(9, 9))
    env = GridWorld(size=5)
    with pytest.raises(ValueError):
        env.step(99)
    with pytest.raises(ValueError):
        env.reset(pos=(9, 9))


def test_action_set_shape():
    assert len(ACTIONS) == 4
