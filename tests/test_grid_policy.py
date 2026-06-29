"""Tests for the TabularQPolicy (TODO.md G.4)."""

import numpy as np

from src.grid.env import ACTIONS
from src.grid.policy import TabularQPolicy


def test_q_table_shape():
    policy = TabularQPolicy(size=8)
    assert policy.Q.shape == (8, 8, len(ACTIONS))


def test_greedy_picks_highest_value_action():
    policy = TabularQPolicy(size=5, epsilon=0.0, seed=0)
    policy.Q[2, 2, 3] = 1.0  # action 3 is best at (2,2)
    assert policy.choose((2, 2)) == 3


def test_update_moves_value_toward_reward():
    policy = TabularQPolicy(size=5, epsilon=0.0, alpha=0.5, gamma=0.0, seed=0)
    before = policy.Q[1, 1, 0]
    policy.update((1, 1), 0, reward=1.0, next_pos=(1, 2))
    assert policy.Q[1, 1, 0] > before


def test_learns_to_prefer_rewarded_cell():
    """With a reward only for stepping onto one cell, the policy learns to go there."""
    policy = TabularQPolicy(size=3, epsilon=0.0, alpha=0.5, gamma=0.9, seed=0)
    # Reward action 3 (right) from (0,0) leading to (1,0); nothing else rewarded.
    for _ in range(50):
        policy.update((0, 0), 3, reward=1.0, next_pos=(1, 0))
        policy.update((0, 0), 0, reward=0.0, next_pos=(0, 0))
    assert policy.choose((0, 0)) == 3


def test_epsilon_one_is_pure_random_but_valid():
    policy = TabularQPolicy(size=4, epsilon=1.0, seed=2)
    actions = {policy.choose((0, 0)) for _ in range(50)}
    assert actions <= set(range(len(ACTIONS)))
    assert len(actions) > 1  # genuinely exploring
