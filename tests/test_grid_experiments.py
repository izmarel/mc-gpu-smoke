"""Experiment-level tests for POC 0 — the core-bet proofs (TODO.md G.5, G.6).

These assert the two things POC 0 exists to prove:

* G.5: pure curiosity explores the grid and then surprise collapses (it learned it).
* G.6 (THE LIE DETECTOR): a raw-surprise agent gets hypnotised by the noise cell;
  a learning-progress agent does not. If this ever fails, the project's core drive
  is broken — that is exactly what we want a regression test to catch.
"""

import pytest

from src.grid.run import (
    REWARD_LEARNING_PROGRESS,
    REWARD_RAW_SURPRISE,
    run_agent,
    run_noise_trap,
)


def test_clean_grid_explores_and_learns():
    """G.5: high coverage + a large drop in surprise == curiosity worked."""
    res = run_agent(size=20, steps=20_000, seed=0)
    assert res.coverage > 0.85, f"coverage too low: {res.coverage:.2%}"
    # Surprise must collapse as the world is learned.
    assert res.mean_error_last < 0.5 * res.mean_error_first


def test_noise_trap_lie_detector():
    """G.6: raw-surprise agent gets stuck on noise; learning-progress agent does not."""
    res = run_noise_trap(size=20, steps=20_000, noise_cell=(10, 10), seed=0)
    control = res["control"]
    real = res["real"]

    fair_share = 1.0 / (20 * 20)  # 0.25%

    # Control is HYPNOTISED: spends a large fraction of all steps on the noise cell.
    assert control.noise_cell_share > 0.05, (
        f"control should get stuck on the noise cell, got {control.noise_cell_share:.2%}"
    )
    # Real agent essentially ignores it (near the fair share, nowhere near stuck).
    assert real.noise_cell_share < 0.02, (
        f"real agent should not linger on noise, got {real.noise_cell_share:.2%}"
    )
    # And the contrast is large.
    assert control.noise_cell_share > 5 * real.noise_cell_share
    # The real agent still has to be a real explorer, not a coward hiding in a corner.
    assert real.coverage > 0.80, f"real agent coverage too low: {real.coverage:.2%}"
    # Sanity: control over-visits the noise cell relative to a fair share.
    assert control.noise_cell_share > 10 * fair_share


@pytest.mark.parametrize("mode", [REWARD_RAW_SURPRISE, REWARD_LEARNING_PROGRESS])
def test_runs_are_deterministic(mode):
    """Same seed -> identical telemetry (reproducible experiments)."""
    a = run_agent(size=10, steps=3_000, reward_mode=mode, noise_cell=(5, 5), seed=1)
    b = run_agent(size=10, steps=3_000, reward_mode=mode, noise_cell=(5, 5), seed=1)
    assert a.noise_cell_visits == b.noise_cell_visits
    assert a.coverage == b.coverage
