"""Unit tests for Phase 7 Wave 4: player-prop legs in the correlation vocabulary.

Covers PLAYER_PROP leg parsing (all formats and error cases), the default
player-leg extension of the stored artifact (line grids, caps, determinism,
logged truncation), read-time complement resolution (NO / UNDER), mixed
team+player joints, and the byte-for-byte Wave 1 regression for team-only
runs.
"""

import base64
import json
import logging
import zlib

import numpy as np
import numpy.typing as npt
import pytest

from simulation_engine.core.correlations import (
    MAX_PLAYER_LEGS,
    CorrelationArtifact,
    UnknownLegError,
    build_correlation_artifact,
    default_legs,
    default_player_legs,
    empirical_joint,
    leg_vector,
)
from simulation_engine.core.params import GameContext, PlayerRates
from simulation_engine.core.plugins import SOCCER_GRID_CONFIG
from simulation_engine.core.plugins.soccer import SoccerParams, SoccerSimulator
from simulation_engine.core.runner import SimulationOutput, run_monte_carlo


def make_output(
    home: list[int],
    away: list[int],
    player_stats: dict[str, dict[str, npt.NDArray[np.int32]]] | None = None,
) -> SimulationOutput:
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
        player_stats=player_stats or {},
    )


def arr(values: list[int]) -> npt.NDArray[np.int32]:
    return np.asarray(values, dtype=np.int32)


def soccer_rates(player_id: str, team: str, goal_share: float) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name=player_id.upper(),
        position="F",
        team=team,  # type: ignore[arg-type]
        rates={"goal_share": goal_share, "shots_per_match": 2.2, "sot_per_match": 1.0, "minutes_share": 0.9},
    )


def run_soccer_with_players(iterations: int = 4000, seed: int = 42) -> SimulationOutput:
    """Small end-to-end soccer run where h1 scores EVERY home goal.

    goal_share=1.0 makes ``h1 scores anytime`` identical to ``home scored``,
    so positive dependence with MONEYLINE:HOME is structural, not statistical.
    """
    sim = SoccerSimulator({})
    sim.set_players([soccer_rates("h1", "HOME", 1.0)], [soccer_rates("a1", "AWAY", 1.0)])
    return run_monte_carlo(
        sim,
        SoccerParams(attack=1.15, defense=0.9, goals_for_per_match=1.6, goals_against_per_match=1.1),
        SoccerParams(attack=0.95, defense=1.05, goals_for_per_match=1.3, goals_against_per_match=1.4),
        GameContext(league="FIFA_WC", neutral_site=True),
        iterations=iterations,
        convergence_threshold=1e-9,  # never stop early
        seed=seed,
        grid_config=SOCCER_GRID_CONFIG,
        capture_players=True,
    )


class TestPlayerLegParsing:
    OUTPUT = make_output(
        [2, 1, 0, 3],
        [1, 1, 1, 0],
        player_stats={
            "p1": {
                "player_shots": arr([3, 2, 0, 4]),
                "player_goal_scorer_anytime": arr([1, 0, 0, 2]),
            }
        },
    )

    def test_over_and_under_vectors(self) -> None:
        assert leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_shots:OVER:2.5").tolist() == [True, False, False, True]
        assert leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_shots:UNDER:2.5").tolist() == [False, True, True, False]

    def test_integer_line_is_strict_both_sides(self) -> None:
        # values == line satisfies neither side (push exclusion, like team legs).
        assert leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_shots:OVER:2").tolist() == [True, False, False, True]
        assert leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_shots:UNDER:2").tolist() == [False, False, True, False]

    def test_yes_and_no_vectors(self) -> None:
        yes = leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_goal_scorer_anytime:YES")
        no = leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_goal_scorer_anytime:NO")
        assert yes.tolist() == [True, False, False, True]
        assert no.tolist() == [False, True, True, False]
        assert np.array_equal(no, ~yes)

    @pytest.mark.parametrize(
        "bad_leg",
        [
            "PLAYER_PROP",  # no components
            "PLAYER_PROP:p1",  # too few parts
            "PLAYER_PROP:p1:player_shots",  # missing side
            "PLAYER_PROP:p1:player_shots:OVER",  # missing line
            "PLAYER_PROP:p1:player_shots:OVER:abc",  # non-numeric line
            "PLAYER_PROP:p1:player_shots:OVER:inf",  # non-finite line
            "PLAYER_PROP:p1:player_shots:MAYBE:2.5",  # unknown side
            "PLAYER_PROP:p1:player_shots:YES",  # YES on an over/under stat
            "PLAYER_PROP:p1:player_shots:NO",  # NO on an over/under stat
            "PLAYER_PROP:p1:player_goal_scorer_anytime:OVER:0.5",  # OVER on a YES/NO stat
            "PLAYER_PROP:p1:player_goal_scorer_anytime:UNDER:0.5",  # UNDER on a YES/NO stat
            "PLAYER_PROP:p1:player_goal_scorer_anytime:YES:1.5",  # YES takes no line
            "PLAYER_PROP::player_shots:OVER:2.5",  # empty player id
            "PLAYER_PROP:p1::OVER:2.5",  # empty stat key
            "PLAYER_PROP:p1:player_shots:OVER:2.5:extra",  # trailing parts
        ],
    )
    def test_malformed_keys_raise(self, bad_leg: str) -> None:
        with pytest.raises(UnknownLegError):
            leg_vector(self.OUTPUT, bad_leg)

    def test_unknown_player_raises_with_clear_message(self) -> None:
        with pytest.raises(UnknownLegError, match="player 'ghost' has no captured stats"):
            leg_vector(self.OUTPUT, "PLAYER_PROP:ghost:player_shots:OVER:2.5")

    def test_unknown_stat_raises_with_captured_stats_listed(self) -> None:
        with pytest.raises(UnknownLegError, match="'player_points' was not captured .* player_shots"):
            leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_points:OVER:20.5")

    def test_side_stat_mismatch_messages_are_actionable(self) -> None:
        with pytest.raises(UnknownLegError, match="YES applies only to YES/NO stats"):
            leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_shots:YES")
        with pytest.raises(UnknownLegError, match="settles YES/NO and takes no line"):
            leg_vector(self.OUTPUT, "PLAYER_PROP:p1:player_goal_scorer_anytime:OVER:0.5")

    def test_uncaptured_props_raise_on_any_player_leg(self) -> None:
        team_only = make_output([2, 1], [1, 1])
        with pytest.raises(UnknownLegError, match="include_player_props"):
            leg_vector(team_only, "PLAYER_PROP:p1:player_shots:OVER:2.5")


class TestDefaultPlayerLegs:
    def test_yes_no_stat_contributes_single_yes_leg(self) -> None:
        output = make_output([1, 2], [0, 1], player_stats={"p1": {"player_goal_scorer_anytime": arr([1, 0])}})
        assert default_player_legs(output) == ["PLAYER_PROP:p1:player_goal_scorer_anytime:YES"]

    def test_over_under_stat_capped_to_three_lines_closest_to_mean(self) -> None:
        # mean 4.2: default_line_grid gives 1.5..7.5; the 3 closest are 3.5/4.5/5.5.
        output = make_output([1] * 5, [0] * 5, player_stats={"p1": {"player_shots": arr([4, 4, 4, 4, 5])}})
        assert default_player_legs(output) == [
            "PLAYER_PROP:p1:player_shots:OVER:3.5",
            "PLAYER_PROP:p1:player_shots:OVER:4.5",
            "PLAYER_PROP:p1:player_shots:OVER:5.5",
        ]

    def test_low_mean_drops_negative_lines_and_keeps_grid_alignment(self) -> None:
        # mean 0.4: the shared grid is 0.5/1.5/2.5/3.5 (negatives dropped); cap keeps 0.5/1.5/2.5.
        output = make_output([1] * 5, [0] * 5, player_stats={"p1": {"player_shots": arr([0, 0, 0, 0, 2])}})
        assert default_player_legs(output) == [
            "PLAYER_PROP:p1:player_shots:OVER:0.5",
            "PLAYER_PROP:p1:player_shots:OVER:1.5",
            "PLAYER_PROP:p1:player_shots:OVER:2.5",
        ]

    def make_many_players(self, n_players: int) -> SimulationOutput:
        values = arr([2, 3, 1, 4])
        stats = {f"p{i:02d}": {"player_shots": values, "player_shots_on_target": values} for i in range(n_players)}
        return make_output([2, 1, 0, 3], [1, 1, 1, 0], player_stats=stats)

    def test_total_cap_is_deterministic_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        # 25 players x 2 over/under stats x 3 lines = 150 candidates -> capped at 120.
        output = self.make_many_players(25)
        with caplog.at_level(logging.WARNING, logger="simulation_engine.core.correlations"):
            first = default_player_legs(output)
            second = default_player_legs(output)
        assert len(first) == MAX_PLAYER_LEGS
        assert first == second  # deterministic: sorted players, stats, lines
        assert first == sorted(first)  # player ids ascending, stats/lines ascending within
        assert "capped at 120" in caplog.text
        assert "dropping 30 of 150" in caplog.text

    def test_no_truncation_log_when_under_cap(self, caplog: pytest.LogCaptureFixture) -> None:
        output = self.make_many_players(20)  # exactly 120 legs
        with caplog.at_level(logging.WARNING, logger="simulation_engine.core.correlations"):
            legs = default_player_legs(output)
        assert len(legs) == MAX_PLAYER_LEGS
        assert "capped" not in caplog.text


@pytest.fixture(scope="module")
def output() -> SimulationOutput:
    return run_soccer_with_players()


@pytest.fixture(scope="module")
def artifact(output: SimulationOutput) -> CorrelationArtifact:
    return build_correlation_artifact(output, include_draw=True)


class TestArtifactWithPlayerLegs:
    def test_player_legs_appended_after_team_legs(
        self, output: SimulationOutput, artifact: CorrelationArtifact
    ) -> None:
        team_legs = default_legs(output, include_draw=True)
        assert artifact.legs[: len(team_legs)] == team_legs
        player_legs = artifact.legs[len(team_legs) :]
        assert player_legs == default_player_legs(output)
        assert "PLAYER_PROP:h1:player_goal_scorer_anytime:YES" in player_legs
        assert all(leg.startswith("PLAYER_PROP:") for leg in player_legs)

    def test_goal_scorer_and_home_moneyline_positively_dependent(self, artifact: CorrelationArtifact) -> None:
        legs = ["MONEYLINE:HOME", "PLAYER_PROP:h1:player_goal_scorer_anytime:YES"]
        marginals, matrix, joint = artifact.subset(legs)
        product = marginals[legs[0]] * marginals[legs[1]]
        # h1 scores every home goal, so home win implies h1 scored: the joint
        # equals P(home win) and exceeds the independence product by
        # P(home win) * P(home scoreless) — far more than the 0.05 tolerance.
        assert joint > product + 0.05
        assert matrix[0][1] > 0.1
        assert joint == pytest.approx(marginals["MONEYLINE:HOME"], abs=1e-9)

    def test_yes_plus_no_marginals_conserve_to_one(self, artifact: CorrelationArtifact) -> None:
        yes, _, _ = artifact.subset(["PLAYER_PROP:h1:player_goal_scorer_anytime:YES"])
        no, _, _ = artifact.subset(["PLAYER_PROP:h1:player_goal_scorer_anytime:NO"])
        yes_leg = "PLAYER_PROP:h1:player_goal_scorer_anytime:YES"
        total = yes[yes_leg] + no[yes_leg[:-3] + "NO"]
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_mixed_leg_joint_equals_direct_and_computation(
        self, output: SimulationOutput, artifact: CorrelationArtifact
    ) -> None:
        legs = ["MONEYLINE:HOME", "TOTAL:OVER:2.5", "PLAYER_PROP:h1:player_goal_scorer_anytime:YES"]
        restored = CorrelationArtifact.from_payload(artifact.to_payload())
        _, _, joint = restored.subset(legs)
        assert joint == empirical_joint(output, legs)
        assert joint > 0.0

    def test_under_half_line_resolves_as_exact_complement(
        self, output: SimulationOutput, artifact: CorrelationArtifact
    ) -> None:
        over_leg = next(leg for leg in artifact.legs if ":player_shots:OVER:" in leg)
        under_leg = over_leg.replace(":OVER:", ":UNDER:")
        marginals, _, joint = artifact.subset([under_leg])
        assert marginals[under_leg] == pytest.approx(1.0 - artifact.marginals[over_leg], abs=1e-9)
        assert joint == empirical_joint(output, [under_leg])

    def test_no_complement_joint_matches_direct(self, output: SimulationOutput, artifact: CorrelationArtifact) -> None:
        legs = ["MONEYLINE:AWAY", "PLAYER_PROP:h1:player_goal_scorer_anytime:NO"]
        _, _, joint = artifact.subset(legs)
        assert joint == empirical_joint(output, legs)

    def test_integer_line_player_complement_not_resolved(self, artifact: CorrelationArtifact) -> None:
        with pytest.raises(UnknownLegError):
            artifact.subset(["PLAYER_PROP:h1:player_shots:UNDER:2"])

    def test_unknown_player_leg_on_player_artifact_raises(self, artifact: CorrelationArtifact) -> None:
        with pytest.raises(UnknownLegError):
            artifact.subset(["PLAYER_PROP:ghost:player_shots:OVER:2.5"])

    def test_explicit_legs_do_not_gain_player_extension(self, output: SimulationOutput) -> None:
        explicit = build_correlation_artifact(output, legs=["MONEYLINE:HOME"], include_draw=True)
        assert explicit.legs == ["MONEYLINE:HOME"]


class TestTeamOnlyRegression:
    """Team-only runs must produce artifacts byte-for-byte identical to Wave 1 behavior."""

    def scores(self) -> tuple[list[int], list[int]]:
        rng = np.random.default_rng(11)
        return rng.integers(0, 6, size=1000).tolist(), rng.integers(0, 6, size=1000).tolist()

    def test_no_player_legs_and_deterministic_payload(self) -> None:
        home, away = self.scores()
        output = make_output(home, away)
        artifact = build_correlation_artifact(output, include_draw=True)
        assert artifact.legs == default_legs(output, include_draw=True)
        assert not any(leg.startswith("PLAYER_PROP:") for leg in artifact.legs)
        rebuilt = build_correlation_artifact(make_output(home, away), include_draw=True)
        assert json.dumps(artifact.to_payload(), sort_keys=True) == json.dumps(rebuilt.to_payload(), sort_keys=True)

    def test_team_block_unchanged_when_player_legs_are_added(self) -> None:
        home, away = self.scores()
        team_only = build_correlation_artifact(make_output(home, away), include_draw=True)
        stats = {"p1": {"player_goal_scorer_anytime": arr(home), "player_shots": arr([h + 1 for h in home])}}
        with_players = build_correlation_artifact(make_output(home, away, player_stats=stats), include_draw=True)
        n = len(team_only.legs)
        assert with_players.legs[:n] == team_only.legs
        assert {leg: with_players.marginals[leg] for leg in team_only.legs} == team_only.marginals
        assert [row[:n] for row in with_players.matrix[:n]] == team_only.matrix
        # Rows are padded to whole bytes independently, so the team rows of
        # the packed matrix are bit-identical.
        row_bytes = (team_only.iterations + 7) // 8
        assert with_players.packed_matrix[: n * row_bytes] == team_only.packed_matrix

    def test_player_leg_request_on_team_only_artifact_gives_clear_error(self) -> None:
        home, away = self.scores()
        artifact = build_correlation_artifact(make_output(home, away), include_draw=True)
        with pytest.raises(UnknownLegError, match="captured no player props"):
            artifact.subset(["MONEYLINE:HOME", "PLAYER_PROP:p1:player_goal_scorer_anytime:YES"])


class TestArtifactSize:
    def test_compressed_blob_stays_reasonable_at_full_scale(self) -> None:
        """Size guard: ~120 player legs at 10k iterations must compress to a modest blob."""
        n = 10_000
        rng = np.random.default_rng(3)
        home = rng.integers(0, 5, size=n).tolist()
        away = rng.integers(0, 5, size=n).tolist()
        values = rng.poisson(2.4, size=(25, 2, n)).astype(np.int32)
        stats = {f"p{i:02d}": {"player_shots": values[i][0], "player_shots_on_target": values[i][1]} for i in range(25)}
        artifact = build_correlation_artifact(make_output(home, away, player_stats=stats), include_draw=True)
        n_legs = len(artifact.legs)
        assert n_legs == 7 + MAX_PLAYER_LEGS  # 2 ML + DRAW + 2 spreads + 2 totals + capped player legs
        assert len(artifact.packed_matrix) == n_legs * ((n + 7) // 8)
        # Compress exactly the way SimulationCache stores the blob (zlib + b64).
        blob = base64.b64encode(zlib.compress(json.dumps({"simulation_run_id": "x", **artifact.to_payload()}).encode()))
        assert len(blob) < 300_000  # ~200KB expected; hard-fail well before Redis pain
