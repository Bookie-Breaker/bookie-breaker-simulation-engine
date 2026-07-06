"""Shared bisection calibrator tests, including basketball refactor parity.

The calibrator was extracted from the basketball plugin in Phase 6 Wave 2;
the parity test replays the original pre-extraction loop and asserts the
shared helper reproduces its PMFs bit-for-bit, so basketball behavior (and
its seeded statistical tests) are provably unchanged.
"""

import numpy as np
import pytest

from simulation_engine.core.calibrate import calibrate_distribution
from simulation_engine.core.plugins.basketball import _build_possession_pmf


def original_basketball_loop(offense, defense, target_ppp: float) -> np.ndarray:
    """The pre-Wave-2 _calibrated_model bisection, verbatim."""
    low, high = 0.5, 1.6
    pmf = _build_possession_pmf(offense, defense)
    for _ in range(40):
        mid = (low + high) / 2
        pmf = _build_possession_pmf(offense, defense, make_scale=mid)
        expected = float(np.dot(pmf, np.arange(len(pmf))))
        if abs(expected - target_ppp) < 1e-5:
            break
        if expected < target_ppp:
            low = mid
        else:
            high = mid
    return pmf


class TestBasketballParity:
    @pytest.mark.parametrize("target_ppp", [0.95, 1.05, 1.13, 1.25])
    def test_bit_for_bit_parity_with_pre_extraction_loop(self, make_team_params, target_ppp: float) -> None:
        offense = make_team_params("o", off_rating=118.0)
        defense = make_team_params("d", def_rating=109.0)
        expected = original_basketball_loop(offense, defense, target_ppp)
        actual = calibrate_distribution(
            lambda scale: _build_possession_pmf(offense, defense, make_scale=scale),
            target_ppp,
            (0.5, 1.6),
        )
        assert np.array_equal(actual, expected)

    def test_defaults_match_original_loop_constants(self, make_team_params) -> None:
        # Default kwargs (40 iterations, 1e-5 tolerance) are the original
        # basketball loop's constants; hitting the target confirms them.
        offense, defense = make_team_params("o"), make_team_params("d")
        pmf = calibrate_distribution(
            lambda scale: _build_possession_pmf(offense, defense, make_scale=scale), 1.1, (0.5, 1.6)
        )
        assert float(np.dot(pmf, np.arange(len(pmf)))) == pytest.approx(1.1, abs=1e-5)


class TestGenericCalibration:
    @staticmethod
    def scaled_pmf(scale: float) -> np.ndarray:
        """Monotone family over 0..3: mean = 2*scale / (0.1 + 0.9*scale), max ~2.22."""
        base = np.array([0.1, 0.2, 0.3, 0.4])
        weights = base * np.array([1.0, scale, scale, scale])
        pmf: np.ndarray = weights / weights.sum()
        return pmf

    @pytest.mark.parametrize("target", [0.5, 1.0, 1.5, 2.0, 2.2])
    def test_hits_target_means_across_range(self, target: float) -> None:
        pmf = calibrate_distribution(self.scaled_pmf, target, (0.01, 50.0), max_iterations=80, tolerance=1e-9)
        assert float(np.dot(pmf, np.arange(4))) == pytest.approx(target, abs=1e-6)
        assert pmf.sum() == pytest.approx(1.0)

    def test_unreachable_target_converges_to_boundary(self) -> None:
        # Max achievable mean at scale=50 is ~2.4; ask for 3.0.
        pmf = calibrate_distribution(self.scaled_pmf, 3.0, (0.01, 50.0), max_iterations=80, tolerance=1e-9)
        boundary = self.scaled_pmf(50.0)
        assert float(np.dot(pmf, np.arange(4))) == pytest.approx(float(np.dot(boundary, np.arange(4))), abs=1e-6)

    def test_rejects_nonpositive_iteration_budget(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            calibrate_distribution(self.scaled_pmf, 1.0, (0.01, 50.0), max_iterations=0)
