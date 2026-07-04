"""Convergence tracker tests."""

import numpy as np
import pytest

from simulation_engine.core.convergence import ConvergenceTracker


def as_arrays(margins: list[int]) -> tuple[np.ndarray, np.ndarray]:
    m = np.array(margins, dtype=np.int32)
    totals = np.abs(m) + 200
    return m, totals.astype(np.int32)


class TestStandardErrorCriterion:
    def test_se_matches_hand_computation(self) -> None:
        tracker = ConvergenceTracker(se_threshold=1e-9)
        margins, totals = as_arrays([2, 4, 6, 8])
        state = tracker.check(margins, totals)
        expected = float(np.std(margins, ddof=1) / np.sqrt(4))
        assert state.standard_error == pytest.approx(expected)

    def test_no_convergence_below_min_iterations(self) -> None:
        tracker = ConvergenceTracker(se_threshold=1000.0, min_iterations=2000)
        margins, totals = as_arrays([1, 2, 3] * 100)
        assert not tracker.check(margins, totals).converged

    def test_se_triggers_when_below_threshold(self) -> None:
        tracker = ConvergenceTracker(se_threshold=1.0, min_iterations=100)
        rng = np.random.default_rng(0)
        margins = np.rint(rng.normal(3, 10, 2000)).astype(np.int32)
        totals = (margins + 210).astype(np.int32)
        # SE = 10 / sqrt(2000) ~ 0.22 < 1.0
        assert tracker.check(margins, totals).converged


class TestStabilityCriterion:
    def test_requires_two_consecutive_quiet_checks(self) -> None:
        tracker = ConvergenceTracker(se_threshold=1e-9, min_iterations=100)
        rng = np.random.default_rng(1)
        base = np.rint(rng.normal(3, 12, 6000)).astype(np.int32)
        totals = (np.abs(base) + 210).astype(np.int32)

        # First check: establishes reference lines and first probabilities
        assert not tracker.check(base[:2000], totals[:2000]).converged
        # Second check on nearly identical data: first quiet check
        assert not tracker.check(base[:2001], totals[:2001]).converged
        # Third check: second consecutive quiet check -> converged
        assert tracker.check(base[:2002], totals[:2002]).converged

    def test_unstable_probabilities_reset_the_streak(self) -> None:
        tracker = ConvergenceTracker(se_threshold=1e-9, min_iterations=10)
        wins, totals_w = as_arrays([10] * 100)
        losses, totals_l = as_arrays([10] * 100 + [-10] * 80)
        tracker.check(wins, totals_w)
        tracker.check(losses, totals_l)  # big probability swing
        state = tracker.check(losses, totals_l)
        # only one quiet check so far, not converged yet
        assert not state.converged
