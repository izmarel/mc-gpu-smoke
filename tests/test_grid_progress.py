"""Tests for the LearningProgress signal (TODO.md G.3)."""

from src.grid.progress import LearningProgress


def test_no_signal_before_min_samples():
    lp = LearningProgress(window=20, min_samples=4)
    lp.record((0, 0), 1.0)
    lp.record((0, 0), 1.0)
    assert lp.progress((0, 0)) == 0.0  # not enough data yet


def test_decreasing_error_gives_positive_progress():
    """Error trending DOWN over time == the system is learning == positive."""
    lp = LearningProgress(window=10, min_samples=4)
    for e in [1.0, 0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1]:
        lp.record((1, 1), e)
    assert lp.progress((1, 1)) > 0.0


def test_flat_high_error_gives_near_zero_progress():
    """The noisy-TV case: permanently high, flat error -> ~0 progress -> boring."""
    lp = LearningProgress(window=10, min_samples=4)
    for _ in range(10):
        lp.record((2, 2), 0.8)
    assert abs(lp.progress((2, 2))) < 1e-6


def test_rising_error_gives_negative_progress():
    lp = LearningProgress(window=10, min_samples=4)
    for e in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
        lp.record((3, 3), e)
    assert lp.progress((3, 3)) < 0.0


def test_unseen_cell_is_zero():
    lp = LearningProgress()
    assert lp.progress((9, 9)) == 0.0
    assert lp.mean_error((9, 9)) == 0.0
