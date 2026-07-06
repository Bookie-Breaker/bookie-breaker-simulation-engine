"""Tests for SimulationOutput -> API contract shape mapping."""

import numpy as np
import pytest

from simulation_engine.core.output import build_distributions, build_result
from simulation_engine.core.runner import SimulationOutput


def make_output(seed: int = 0, n: int = 4000) -> SimulationOutput:
    rng = np.random.default_rng(seed)
    home = np.rint(rng.normal(112, 9, n)).astype(np.int32)
    away = np.rint(rng.normal(109, 9, n)).astype(np.int32)
    tied = home == away
    home[tied] += 1
    margins = (home - away).astype(np.int32)
    totals = (home + away).astype(np.int32)
    margin_mean = float(np.mean(margins))
    total_mean = float(np.mean(totals))
    spread_lines = [-round(margin_mean) + k + 0.5 for k in range(-4, 4)]
    total_lines = [round(total_mean) + k + 0.5 for k in range(-4, 4)]
    return SimulationOutput(
        iterations_run=n,
        converged=True,
        convergence_iteration=n,
        standard_error=0.2,
        home_scores=home,
        away_scores=away,
        margins=margins,
        totals=totals,
        home_win_prob=float(np.mean(margins > 0)),
        away_win_prob=float(np.mean(margins < 0)),
        draw_prob=0.0,
        margin_mean=margin_mean,
        margin_std=float(np.std(margins, ddof=1)),
        total_mean=total_mean,
        total_std=float(np.std(totals, ddof=1)),
        spread_covers={line: float(np.mean(margins > -line)) for line in spread_lines},
        total_overs={line: float(np.mean(totals > line)) for line in total_lines},
        elapsed_ms=50.0,
    )


class TestBuildResult:
    def test_spread_keys_are_signed_half_lines(self) -> None:
        result = build_result(make_output(), "res-1")
        for key in result.spread_cover_probabilities:
            assert key[0] in "+-"
            assert key.endswith(".5")

    def test_total_keys_are_unsigned_half_lines(self) -> None:
        result = build_result(make_output(), "res-1")
        for key in result.total_over_probabilities:
            assert not key.startswith("+")
            assert key.endswith(".5")

    def test_cover_probability_increases_with_handicap(self) -> None:
        # More points given to the home side -> easier to cover:
        # P(margin > -h) is nondecreasing in h
        covers = build_result(make_output(), "res-1").spread_cover_probabilities
        ordered = [covers[key] for key in sorted(covers, key=float)]
        assert all(a <= b for a, b in zip(ordered, ordered[1:], strict=False))

    def test_probabilities_rounded_and_bounded(self) -> None:
        result = build_result(make_output(), "res-1")
        assert 0.0 <= result.home_win_probability <= 1.0
        assert result.home_win_probability + result.away_win_probability + result.draw_probability == pytest.approx(
            1.0, abs=1e-3
        )

    def test_percentiles_ordered(self) -> None:
        result = build_result(make_output(), "res-1")
        for dist in (result.percentiles.margin, result.percentiles.total):
            values = [dist[p] for p in ("10", "25", "50", "75", "90")]
            assert values == sorted(values)


class TestPushSerialization:
    def test_push_maps_default_empty(self) -> None:
        result = build_result(make_output(), "res-1")
        assert result.spread_push_probabilities == {}
        assert result.total_push_probabilities == {}

    def test_integer_push_keys_signed_and_rounded(self) -> None:
        output = make_output()
        output.spread_pushes = {-3.0: 0.06789, 2.0: 0.05}
        output.total_pushes = {220.0: 0.04321}
        result = build_result(output, "res-1")
        assert result.spread_push_probabilities == {"-3.0": 0.0679, "+2.0": 0.05}
        assert result.total_push_probabilities == {"220.0": 0.0432}


class TestBuildDistributions:
    def test_all_four_distributions_present(self) -> None:
        distributions = build_distributions(make_output())
        assert set(distributions) == {"home_score", "away_score", "margin", "total"}

    def test_frequencies_sum_to_one(self) -> None:
        for dist in build_distributions(make_output()).values():
            assert sum(dist.values.values()) == pytest.approx(1.0, abs=0.01)
            assert dist.min <= dist.max
