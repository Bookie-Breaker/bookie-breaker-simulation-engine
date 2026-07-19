"""Unit tests for the Phase 7 Wave 1 same-game parlay correlation artifact."""

import math

import numpy as np
import pytest

from simulation_engine.core.correlations import (
    CorrelationArtifact,
    UnknownLegError,
    build_correlation_artifact,
    default_legs,
    empirical_joint,
    format_line,
    leg_vector,
)
from simulation_engine.core.params import GameContext
from simulation_engine.core.plugins import SOCCER_GRID_CONFIG, get_plugin
from simulation_engine.core.plugins.soccer import SoccerParams, SoccerSimulator
from simulation_engine.core.runner import SimulationOutput, run_monte_carlo


def make_output(home: list[int], away: list[int]) -> SimulationOutput:
    """Build a SimulationOutput directly from score arrays (unit-test shim)."""
    home_arr = np.asarray(home, dtype=np.int32)
    away_arr = np.asarray(away, dtype=np.int32)
    margins = home_arr - away_arr
    totals = home_arr + away_arr
    return SimulationOutput(
        iterations_run=len(home_arr),
        converged=True,
        convergence_iteration=None,
        standard_error=0.0,
        home_scores=home_arr,
        away_scores=away_arr,
        margins=margins,
        totals=totals,
        home_win_prob=float(np.mean(margins > 0)),
        away_win_prob=float(np.mean(margins < 0)),
        draw_prob=float(np.mean(margins == 0)),
        margin_mean=float(np.mean(margins)),
        margin_std=float(np.std(margins)) or 1.0,
        total_mean=float(np.mean(totals)),
        total_std=float(np.std(totals)) or 1.0,
        spread_covers={-1.5: float(np.mean(margins > 1.5)), 0.5: float(np.mean(margins > -0.5))},
        total_overs={2.5: float(np.mean(totals > 2.5)), 3.5: float(np.mean(totals > 3.5))},
    )


class TestLegVector:
    def test_moneyline_sides(self) -> None:
        # margins: +2, 0, -1  -> home win, draw, away win
        output = make_output([3, 1, 0], [1, 1, 1])
        assert leg_vector(output, "MONEYLINE:HOME").tolist() == [True, False, False]
        assert leg_vector(output, "MONEYLINE:AWAY").tolist() == [False, False, True]
        assert leg_vector(output, "MONEYLINE:DRAW").tolist() == [False, True, False]

    def test_spread_home_semantics_and_push_exclusion(self) -> None:
        # margins: 3, 2, 1. HOME line -2 covers when margin > 2; margin == 2 is a push -> False.
        output = make_output([3, 2, 1], [0, 0, 0])
        assert leg_vector(output, "SPREAD:HOME:-2").tolist() == [True, False, False]
        # AWAY line is the negation: away +2 covers when margin < 2; the push is False on BOTH sides.
        assert leg_vector(output, "SPREAD:AWAY:2").tolist() == [False, False, True]

    def test_spread_half_point_lines(self) -> None:
        output = make_output([3, 2, 1], [0, 0, 0])
        assert leg_vector(output, "SPREAD:HOME:-1.5").tolist() == [True, True, False]
        assert leg_vector(output, "SPREAD:AWAY:1.5").tolist() == [False, False, True]

    def test_total_push_exclusion(self) -> None:
        # totals: 4, 3, 2 with integer line 3: the total == 3 iteration is False for OVER and UNDER.
        output = make_output([2, 2, 1], [2, 1, 1])
        assert leg_vector(output, "TOTAL:OVER:3").tolist() == [True, False, False]
        assert leg_vector(output, "TOTAL:UNDER:3").tolist() == [False, False, True]
        assert leg_vector(output, "TOTAL:OVER:2.5").tolist() == [True, True, False]

    @pytest.mark.parametrize(
        "bad_leg",
        [
            "PROP:HOME:1.5",  # unknown market
            "MONEYLINE:NEITHER",  # unknown side
            "MONEYLINE:HOME:1.5",  # moneyline takes no line
            "SPREAD:OVER:1.5",  # side from the wrong market
            "SPREAD:HOME",  # missing line
            "TOTAL:OVER:abc",  # non-numeric line
            "TOTAL:OVER:inf",  # non-finite line
            "",
        ],
    )
    def test_unknown_keys_raise(self, bad_leg: str) -> None:
        output = make_output([1, 2], [0, 1])
        with pytest.raises(UnknownLegError):
            leg_vector(output, bad_leg)


class TestDefaultLegs:
    def test_grid_and_moneylines_with_exact_formats(self) -> None:
        output = make_output([2, 1], [0, 1])
        legs = default_legs(output)
        assert legs == [
            "MONEYLINE:HOME",
            "MONEYLINE:AWAY",
            "SPREAD:HOME:-1.5",
            "SPREAD:HOME:0.5",
            "TOTAL:OVER:2.5",
            "TOTAL:OVER:3.5",
        ]

    def test_draw_included_only_when_requested(self) -> None:
        output = make_output([2, 1], [0, 1])
        assert "MONEYLINE:DRAW" not in default_legs(output)
        assert "MONEYLINE:DRAW" in default_legs(output, include_draw=True)

    def test_format_line_uses_percent_g(self) -> None:
        assert format_line(-1.5) == "-1.5"
        assert format_line(2.5) == "2.5"
        assert format_line(220.0) == "220"
        assert format_line(-0.0) == "0"


class TestBuildCorrelationArtifact:
    def test_known_phi_recovered(self) -> None:
        # 2x2 design over (home ML, over 1.5): counts n11=30, n10=10, n01=10, n00=50.
        pairs = [(3, 0)] * 30 + [(1, 0)] * 10 + [(0, 3)] * 10 + [(0, 1)] * 50
        home, away = zip(*pairs, strict=True)
        output = make_output(list(home), list(away))
        legs = ["MONEYLINE:HOME", "TOTAL:OVER:1.5"]
        artifact = build_correlation_artifact(output, legs=legs)
        n11, n10, n01, n00 = 30, 10, 10, 50
        expected_phi = (n11 * n00 - n10 * n01) / math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
        assert artifact.matrix[0][1] == pytest.approx(expected_phi, abs=1e-5)
        assert artifact.marginals["MONEYLINE:HOME"] == pytest.approx(0.4)
        assert artifact.marginals["TOTAL:OVER:1.5"] == pytest.approx(0.4)
        assert artifact.iterations == 100

    def test_matrix_symmetric_unit_diagonal_nan_guarded(self) -> None:
        rng = np.random.default_rng(7)
        home = rng.integers(0, 5, size=500).tolist()
        away = rng.integers(0, 5, size=500).tolist()
        # TOTAL:OVER:-0.5 is always true (totals >= 0): a zero-variance leg.
        legs = ["MONEYLINE:HOME", "MONEYLINE:AWAY", "TOTAL:OVER:2.5", "TOTAL:OVER:-0.5"]
        artifact = build_correlation_artifact(make_output(home, away), legs=legs)
        matrix = np.asarray(artifact.matrix)
        assert np.allclose(matrix, matrix.T)
        assert np.all(np.diagonal(matrix) == 1.0)
        assert np.all(np.isfinite(matrix))
        # Zero-variance leg correlates 0.0 with every other leg.
        assert matrix[3, :3].tolist() == [0.0, 0.0, 0.0]

    def test_perfectly_dependent_legs(self) -> None:
        output = make_output([3, 3, 0, 0], [0, 0, 3, 3])
        artifact = build_correlation_artifact(output, legs=["MONEYLINE:HOME", "SPREAD:HOME:-0.5", "MONEYLINE:AWAY"])
        assert artifact.matrix[0][1] == pytest.approx(1.0)
        assert artifact.matrix[0][2] == pytest.approx(-1.0)


class TestEmpiricalJoint:
    def test_independent_legs_joint_equals_product_of_marginals(self) -> None:
        # Cross-product design: winner and total-band vary independently.
        # win/high (3,1), win/low (1,0), lose/high (1,3), lose/low (0,1); line 2.5.
        pairs = [(3, 1)] * 6 + [(1, 0)] * 6 + [(1, 3)] * 4 + [(0, 1)] * 4
        home, away = zip(*pairs, strict=True)
        output = make_output(list(home), list(away))
        legs = ["MONEYLINE:HOME", "TOTAL:OVER:2.5"]
        joint = empirical_joint(output, legs)
        product = float(np.mean(leg_vector(output, legs[0]))) * float(np.mean(leg_vector(output, legs[1])))
        assert joint == pytest.approx(product)
        assert joint == pytest.approx(0.3)

    def test_empty_legs_raise(self) -> None:
        with pytest.raises(UnknownLegError):
            empirical_joint(make_output([1], [0]), [])


class TestPackedMatrixRoundTrip:
    def make_artifact_and_output(self) -> tuple[CorrelationArtifact, SimulationOutput]:
        rng = np.random.default_rng(11)
        home = rng.integers(0, 6, size=1000).tolist()
        away = rng.integers(0, 6, size=1000).tolist()
        output = make_output(home, away)
        return build_correlation_artifact(output, include_draw=True), output

    def test_payload_round_trip(self) -> None:
        artifact, _ = self.make_artifact_and_output()
        restored = CorrelationArtifact.from_payload(artifact.to_payload())
        assert restored == artifact

    def test_subset_joint_matches_direct_joint(self) -> None:
        artifact, output = self.make_artifact_and_output()
        legs = ["MONEYLINE:HOME", "SPREAD:HOME:0.5", "TOTAL:OVER:3.5"]
        restored = CorrelationArtifact.from_payload(artifact.to_payload())
        _, _, joint = restored.subset(legs)
        assert joint == empirical_joint(output, legs)

    def test_subset_resolves_half_point_complements(self) -> None:
        artifact, output = self.make_artifact_and_output()
        # SPREAD:AWAY:-0.5 is the exact complement of stored SPREAD:HOME:0.5;
        # TOTAL:UNDER:2.5 the complement of stored TOTAL:OVER:2.5.
        legs = ["SPREAD:AWAY:-0.5", "TOTAL:UNDER:2.5"]
        marginals, matrix, joint = artifact.subset(legs)
        assert joint == empirical_joint(output, legs)
        assert marginals["SPREAD:AWAY:-0.5"] == pytest.approx(1.0 - artifact.marginals["SPREAD:HOME:0.5"])
        # Negating one leg of a pair flips the sign of the stored correlation.
        home_idx = artifact.legs.index("SPREAD:HOME:0.5")
        over_idx = artifact.legs.index("TOTAL:OVER:2.5")
        assert matrix[0][1] == pytest.approx(artifact.matrix[home_idx][over_idx], abs=1e-5)
        assert np.asarray(matrix).diagonal().tolist() == [1.0, 1.0]

    def test_subset_marginals_and_matrix_are_submatrix(self) -> None:
        artifact, _ = self.make_artifact_and_output()
        legs = ["MONEYLINE:HOME", "TOTAL:OVER:3.5"]
        marginals, matrix, _ = artifact.subset(legs)
        i, j = artifact.legs.index(legs[0]), artifact.legs.index(legs[1])
        assert marginals == {leg: artifact.marginals[leg] for leg in legs}
        assert matrix[0][1] == artifact.matrix[i][j]

    def test_unknown_or_unresolvable_legs_raise(self) -> None:
        artifact, _ = self.make_artifact_and_output()
        with pytest.raises(UnknownLegError):
            artifact.subset(["SPREAD:HOME:-99.5"])  # outside the stored grid
        with pytest.raises(UnknownLegError):
            artifact.subset(["SPREAD:AWAY:2"])  # integer-line complement is not exact (pushes)
        with pytest.raises(UnknownLegError):
            artifact.subset(["BOGUS:LEG"])


class TestSoccerJointGrid:
    def make_simulator(self) -> SoccerSimulator:
        spec = get_plugin("FIFA_WC")
        simulator = spec.simulator(dict(spec.plugin_config))
        assert isinstance(simulator, SoccerSimulator)
        home = SoccerParams(attack=1.25, defense=0.85, goals_for_per_match=1.7, goals_against_per_match=1.1)
        away = SoccerParams(attack=1.05, defense=0.95, goals_for_per_match=1.4, goals_against_per_match=1.3)
        simulator.set_parameters(home, away, GameContext(league="FIFA_WC"))
        return simulator

    def test_joint_grid_present_and_normalized(self) -> None:
        grid = self.make_simulator().joint_grid()
        assert grid is not None
        assert grid.shape == (13, 13)
        assert float(grid.sum()) == pytest.approx(1.0)

    def test_analytic_home_win_matches_empirical(self) -> None:
        simulator = self.make_simulator()
        grid = simulator.joint_grid()
        assert grid is not None
        # Rows are home goals: home wins on the strictly-lower triangle.
        analytic_home_win = float(np.tril(grid, k=-1).sum())
        home = SoccerParams(attack=1.25, defense=0.85, goals_for_per_match=1.7, goals_against_per_match=1.1)
        away = SoccerParams(attack=1.05, defense=0.95, goals_for_per_match=1.4, goals_against_per_match=1.3)
        output = run_monte_carlo(
            simulator,
            home,
            away,
            GameContext(league="FIFA_WC"),
            iterations=20_000,
            convergence_threshold=1e-9,  # never stop early
            seed=42,
            grid_config=SOCCER_GRID_CONFIG,
        )
        assert output.home_win_prob == pytest.approx(analytic_home_win, abs=0.02)

    def test_base_simulator_has_no_joint_grid(self) -> None:
        spec = get_plugin("NBA")
        assert spec.simulator(dict(spec.plugin_config)).joint_grid() is None
