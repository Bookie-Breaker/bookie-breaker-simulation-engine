"""Live re-simulation plugin tests (Phase 7 Wave 2).

Covers, per plugin: pregame bit-identity (seeded outputs pinned from main at
commit d00fb80, before this branch — the pregame path must be byte-identical),
continuity (live_state with fraction_remaining=1.0 and a 0-0 score reproduces
the pregame run exactly, since every plugin's live conditioning degenerates to
the pregame model there), dominance (big lead + small remaining fraction
drives the leader's win probability to ~1), score-offset correctness (final
scores never fall below the current score), and the sport-specific live
refinements (football possession split, baseball explicit-state resume with
the run-expectancy partial-inning adjustment).
"""

from dataclasses import replace

import numpy as np
import pytest

from simulation_engine.core.params import GameContext, LiveState, TeamParams
from simulation_engine.core.plugins import get_plugin
from simulation_engine.core.plugins.baseball import BaseballParams, BaseballSimulator
from simulation_engine.core.plugins.football import FootballParams, FootballSimulator
from simulation_engine.core.plugins.hockey import HockeyParams as HkParams
from simulation_engine.core.plugins.hockey import HockeySimulator
from simulation_engine.core.plugins.soccer import SoccerParams, SoccerSimulator

SEED = 1234
PIN_N = 64
N = 20_000


def nba_params(team_id: str, **overrides: float) -> TeamParams:
    defaults = {
        "pace": 100.0,
        "off_rating": 114.0,
        "def_rating": 112.0,
        "three_pct": 0.365,
        "two_pct": 0.54,
        "ft_pct": 0.78,
        "three_attempt_rate": 0.39,
        "ft_rate": 0.26,
        "tov_pct": 13.0,
        "oreb_pct": 27.0,
        "opp_three_pct": 0.36,
        "opp_two_pct": 0.54,
        "opp_ft_rate": 0.26,
        "forced_tov_pct": 13.0,
        "opp_oreb_pct": 27.0,
    }
    defaults.update(overrides)
    return TeamParams(team_id=team_id, abbreviation=team_id.upper()[:3], **defaults)  # type: ignore[arg-type]


PARAMS = {
    "NBA": (nba_params("h"), nba_params("a", off_rating=112.0, def_rating=113.0)),
    "FIFA_WC": (
        SoccerParams(attack=1.15, defense=0.9, goals_for_per_match=1.55, goals_against_per_match=1.2),
        SoccerParams(attack=0.95, defense=1.05, goals_for_per_match=1.28, goals_against_per_match=1.4),
    ),
    "NHL": (
        HkParams(
            goals_for_per_game=3.2,
            goals_against_per_game=2.8,
            power_play_pct=0.22,
            penalty_kill_pct=0.80,
            team_save_pct=0.905,
        ),
        HkParams(
            goals_for_per_game=2.9,
            goals_against_per_game=3.1,
            power_play_pct=0.19,
            penalty_kill_pct=0.78,
            team_save_pct=0.900,
        ),
    ),
    "NFL": (
        FootballParams(
            points_per_game=24.0,
            points_allowed_per_game=20.0,
            drives_per_game=11.0,
            points_per_drive_off=2.2,
            points_per_drive_def=1.8,
            epa_per_play_off=0.0,
            epa_per_play_def=0.0,
        ),
        FootballParams(
            points_per_game=20.0,
            points_allowed_per_game=24.0,
            drives_per_game=10.8,
            points_per_drive_off=1.8,
            points_per_drive_def=2.1,
            epa_per_play_off=0.0,
            epa_per_play_def=0.0,
        ),
    ),
    "MLB": (
        BaseballParams(
            runs_scored_per_game=4.8,
            runs_allowed_per_game=4.2,
            team_era=3.9,
            team_fip=3.8,
            bullpen_era=4.0,
        ),
        BaseballParams(
            runs_scored_per_game=4.3,
            runs_allowed_per_game=4.6,
            team_era=4.2,
            team_fip=4.1,
            bullpen_era=4.3,
        ),
    ),
}

CONTEXTS = {
    "NBA": GameContext(league="NBA"),
    "FIFA_WC": GameContext(league="FIFA_WC", neutral_site=True),
    "NHL": GameContext(league="NHL"),
    "NFL": GameContext(league="NFL"),
    "MLB": GameContext(league="MLB"),
}


def _scores(spec: str) -> list[int]:
    return [int(value) for value in spec.split()]


# Seeded simulate_games(rng(1234), 64) outputs captured on main (d00fb80)
# BEFORE this branch, with the PARAMS/CONTEXTS above. The pregame path must
# reproduce these byte for byte.
PINNED = {
    "NBA": (
        _scores(
            "101 103 106 113 126 113 117 128 103 135 109 107 113 111 117 117 121 139 114 104 116 132 118 106 104 114 "
            "108 150 99 121 106 106 116 110 110 122 109 93 124 119 120 141 123 121 104 109 109 102 107 144 113 119 "
            "101 103 121 100 102 144 112 121 106 130 113 116 "
        ),
        _scores(
            "119 104 117 119 107 128 110 134 88 87 91 121 121 85 96 106 125 133 119 79 111 107 108 95 109 116 125 "
            "104 110 119 104 110 104 108 103 111 104 122 125 123 101 117 109 132 110 100 120 106 103 117 116 89 102 "
            "110 103 96 106 113 98 114 105 119 108 125 "
        ),
    ),
    "FIFA_WC": (
        _scores(
            "5 1 4 1 1 0 1 1 4 1 1 2 3 3 2 2 2 1 0 3 0 2 2 2 0 5 1 2 0 1 3 1 2 4 0 5 0 4 0 1 3 1 3 5 3 1 1 1 0 1 2 2 "
            "1 1 4 2 1 1 2 3 1 2 3 3 "
        ),
        _scores(
            "0 1 0 0 1 1 0 1 2 0 2 1 1 1 1 1 2 0 2 1 0 1 1 1 0 0 2 0 0 0 1 2 2 0 2 3 3 1 1 2 1 0 1 0 1 2 1 3 1 0 0 2 "
            "2 1 1 3 2 2 2 1 1 2 0 2 "
        ),
    ),
    "NHL": (
        _scores(
            "8 3 6 2 2 1 2 2 7 2 4 4 6 6 5 4 5 3 2 6 1 4 4 4 1 8 3 3 0 2 5 3 5 6 2 8 2 7 1 4 5 2 6 8 5 4 3 3 1 2 3 5 "
            "4 2 7 5 3 4 5 5 2 5 5 6 "
        ),
        _scores(
            "0 2 4 3 7 4 3 7 3 3 3 2 0 0 4 3 0 2 1 1 2 5 3 2 2 1 4 6 1 3 7 2 0 4 1 5 1 1 3 3 4 4 1 0 4 3 1 4 3 3 8 0 "
            "3 5 2 1 4 3 1 6 4 0 3 2 "
        ),
    ),
    "NFL": (
        _scores(
            "10 24 31 27 26 37 31 27 27 31 34 20 10 31 24 17 3 31 30 7 35 48 34 41 52 13 34 34 37 26 41 19 48 13 34 "
            "28 17 21 31 21 27 10 9 20 45 24 21 24 38 36 27 24 27 13 48 23 13 20 52 41 19 27 24 20 "
        ),
        _scores(
            "17 21 17 17 23 16 27 17 12 45 27 13 24 30 14 3 13 59 14 3 20 19 10 7 23 24 27 13 12 10 6 27 24 13 10 6 "
            "6 13 6 19 17 19 24 26 10 26 7 27 7 9 20 20 6 26 27 17 30 14 17 28 16 30 23 17 "
        ),
    ),
    "MLB": (
        _scores(
            "6 7 9 2 6 6 9 3 2 1 3 4 3 0 3 4 1 3 10 3 6 7 4 0 5 7 13 2 6 5 0 6 5 2 7 2 0 1 4 2 8 7 6 11 2 7 13 5 1 6 "
            "2 7 6 3 4 8 9 5 3 7 0 3 2 5 "
        ),
        _scores(
            "10 4 8 3 7 4 0 4 7 5 2 7 2 5 1 1 0 1 2 7 0 0 2 5 0 18 2 4 7 4 5 1 4 4 5 6 5 4 2 1 2 0 2 4 7 3 4 3 2 4 4 "
            "9 3 5 3 9 2 1 9 2 3 8 3 3 "
        ),
    ),
}

LEAGUES = list(PINNED)


def make_sim(league: str, context: GameContext):
    spec = get_plugin(league)
    sim = spec.simulator(dict(spec.plugin_config))
    home, away = PARAMS[league]
    sim.set_parameters(home, away, context)
    return sim


def live_context(league: str, home_score: int, away_score: int, fraction: float, **kw) -> GameContext:
    state = LiveState(home_score=home_score, away_score=away_score, fraction_remaining=fraction, **kw)
    return replace(CONTEXTS[league], live_state=state)


def simulate(league: str, context: GameContext, n: int = N, seed: int = SEED):
    return make_sim(league, context).simulate_games(np.random.default_rng(seed), n)


class TestPregameBitIdentity:
    """The pregame path must be byte-identical to main (fixtures pinned pre-branch)."""

    @pytest.mark.parametrize("league", LEAGUES)
    def test_seeded_pregame_outputs_match_main(self, league) -> None:
        home, away = simulate(league, CONTEXTS[league], n=PIN_N)
        pinned_home, pinned_away = PINNED[league]
        assert home.tolist() == pinned_home
        assert away.tolist() == pinned_away


class TestContinuity:
    """fraction_remaining=1.0 with a 0-0 score degenerates to the pregame model.

    Every plugin's conditioning is exact at the boundary (rates x 1.0, counts
    x 1.0 rounded, baseball coarse resume from inning 1), so with a shared
    seed the live run is not merely close — it is bit-identical.
    """

    @pytest.mark.parametrize("league", LEAGUES)
    def test_full_fraction_zero_score_reproduces_pregame_exactly(self, league) -> None:
        pregame_home, pregame_away = simulate(league, CONTEXTS[league])
        live_home, live_away = simulate(league, live_context(league, 0, 0, 1.0))
        assert np.array_equal(pregame_home, live_home)
        assert np.array_equal(pregame_away, live_away)


class TestDominance:
    """A big lead with little time remaining makes the leader ~certain to win."""

    CASES = {
        "NBA": (100, 70, 0.05),
        "FIFA_WC": (3, 0, 0.05),
        "NHL": (4, 0, 0.05),
        "NFL": (28, 0, 0.05),
        "MLB": (8, 0, 0.1),
    }

    @pytest.mark.parametrize("league", LEAGUES)
    def test_leader_win_probability_approaches_one(self, league) -> None:
        home_score, away_score, fraction = self.CASES[league]
        home, away = simulate(league, live_context(league, home_score, away_score, fraction))
        assert float(np.mean(home > away)) > 0.99


class TestScoreOffsets:
    """Final scores are current score + remainder: never below the live score."""

    CASES = {
        "NBA": (55, 48, 0.5),
        "FIFA_WC": (2, 1, 0.4),
        "NHL": (2, 1, 0.4),
        "NFL": (14, 10, 0.5),
        "MLB": (3, 2, 0.4),
    }

    @pytest.mark.parametrize("league", LEAGUES)
    def test_min_scores_and_total_respect_current_score(self, league) -> None:
        home_score, away_score, fraction = self.CASES[league]
        home, away = simulate(league, live_context(league, home_score, away_score, fraction))
        assert int(home.min()) >= home_score
        assert int(away.min()) >= away_score
        assert int((home + away).min()) >= home_score + away_score


class TestSoccerLive:
    def test_lambdas_scale_by_fraction_remaining(self) -> None:
        pregame = make_sim("FIFA_WC", CONTEXTS["FIFA_WC"])
        live = make_sim("FIFA_WC", live_context("FIFA_WC", 1, 0, 0.35))
        assert isinstance(pregame, SoccerSimulator) and isinstance(live, SoccerSimulator)
        assert live._lam_home == pytest.approx(pregame._lam_home * 0.35)
        assert live._lam_away == pytest.approx(pregame._lam_away * 0.35)

    def test_joint_grid_is_shifted_by_current_score(self) -> None:
        live = make_sim("FIFA_WC", live_context("FIFA_WC", 2, 1, 0.4))
        assert isinstance(live, SoccerSimulator)
        grid = live.joint_grid()
        assert grid is not None
        assert grid.shape == (13 + 2, 13 + 1)
        assert grid.sum() == pytest.approx(1.0)
        assert grid[:2, :].sum() == 0.0  # no mass below the current home score
        assert grid[:, :1].sum() == 0.0  # no mass below the current away score
        assert np.array_equal(grid[2:, 1:], live._grid)

    def test_draws_remain_valid_outcomes_when_tied_live(self) -> None:
        home, away = simulate("FIFA_WC", live_context("FIFA_WC", 1, 1, 0.3))
        assert int(np.sum(home == away)) > 0  # regulation soccer keeps draws (ADR-027)


class TestHockeyLive:
    def test_tied_live_game_resolves_via_ot_no_final_ties(self) -> None:
        home, away = simulate("NHL", live_context("NHL", 2, 2, 0.05))
        assert not bool(np.any(home == away))
        assert int(np.sum(home > away)) > 0
        assert int(np.sum(away > home)) > 0
        # With almost no regulation left, most deciders are the OT/SO +1 goal.
        assert float(np.mean(np.abs(home - away) == 1)) > 0.9

    def test_regulation_joint_grid_is_shifted_by_current_score(self) -> None:
        live = make_sim("NHL", live_context("NHL", 2, 1, 0.4))
        assert isinstance(live, HockeySimulator)
        grid = live.joint_grid()
        assert grid is not None
        assert grid.shape == (10 + 2, 10 + 1)
        assert grid.sum() == pytest.approx(1.0)
        assert np.array_equal(grid[2:, 1:], live._grid)


class TestBasketballLive:
    def test_remaining_points_scale_with_fraction(self) -> None:
        pregame_home, pregame_away = simulate("NBA", CONTEXTS["NBA"])
        pregame_total_mean = float(np.mean(pregame_home + pregame_away))
        live_home, live_away = simulate("NBA", live_context("NBA", 55, 48, 0.1))
        added_mean = float(np.mean(live_home + live_away)) - (55 + 48)
        # Remainder possessions are ~10% of the full game, so remainder points
        # should be ~10% of the pregame total mean (OT noise adds slack).
        assert added_mean == pytest.approx(0.1 * pregame_total_mean, rel=0.2)

    def test_tied_live_game_resolves_via_overtime(self) -> None:
        home, away = simulate("NBA", live_context("NBA", 100, 100, 0.01))
        assert not bool(np.any(home == away))


class TestFootballLive:
    def test_possession_gets_the_fractional_drive(self) -> None:
        live = make_sim("NFL", live_context("NFL", 14, 10, 0.33, possession="HOME"))
        assert isinstance(live, FootballSimulator)
        counts = np.array([10, 11], dtype=np.int64)
        home_counts, away_counts = live._remaining_drive_counts(counts)
        assert home_counts.tolist() == [4, 4]  # ceil(3.3), ceil(3.63)
        assert away_counts.tolist() == [3, 3]  # floor(3.3), floor(3.63)

        away_ball = make_sim("NFL", live_context("NFL", 14, 10, 0.33, possession="AWAY"))
        assert isinstance(away_ball, FootballSimulator)
        home_counts, away_counts = away_ball._remaining_drive_counts(counts)
        assert home_counts.tolist() == [3, 3]
        assert away_counts.tolist() == [4, 4]

    def test_no_possession_rounds_both_sides_equally(self) -> None:
        live = make_sim("NFL", live_context("NFL", 14, 10, 0.33))
        assert isinstance(live, FootballSimulator)
        home_counts, away_counts = live._remaining_drive_counts(np.array([10, 11], dtype=np.int64))
        assert home_counts.tolist() == away_counts.tolist() == [3, 4]  # rint(3.3), rint(3.63)

    def test_possession_shifts_the_mean_margin(self) -> None:
        home_ball_h, home_ball_a = simulate("NFL", live_context("NFL", 14, 10, 0.35, possession="HOME"))
        away_ball_h, away_ball_a = simulate("NFL", live_context("NFL", 14, 10, 0.35, possession="AWAY"))
        margin_home_ball = float(np.mean(home_ball_h - home_ball_a))
        margin_away_ball = float(np.mean(away_ball_h - away_ball_a))
        assert margin_home_ball > margin_away_ball + 1.0  # the extra drive is worth ~2 points


class TestBaseballLive:
    def test_bottom_ninth_two_out_tie_resume(self) -> None:
        context = live_context("MLB", 3, 3, 0.02, period=9, half="BOTTOM", outs=2, bases="---")
        home, away = simulate("MLB", context)
        totals = home + away
        assert not bool(np.any(home == away))  # extras resolve remaining ties
        assert int(home.min()) >= 3 and int(away.min()) >= 3
        assert int(totals.min()) == 7  # tightest finish: a single deciding run
        assert float(np.mean(totals)) < 9.5  # short tail: 2-out bottom 9 + extras only
        # The canonical walk-off shape — home 4, away 3 — must be common.
        assert float(np.mean((home == 4) & (away == 3))) > 0.15

    def test_run_expectancy_adjustment_orders_partial_innings(self) -> None:
        loaded = live_context("MLB", 0, 0, 0.3, period=7, half="BOTTOM", outs=0, bases="123")
        empty_two_out = live_context("MLB", 0, 0, 0.3, period=7, half="BOTTOM", outs=2, bases="---")
        loaded_home, _ = simulate("MLB", loaded)
        empty_home, _ = simulate("MLB", empty_two_out)
        # Bases loaded, nobody out is worth ~1 extra expected run vs two out,
        # bases empty; only the in-progress bottom 7 differs between the runs.
        assert float(np.mean(loaded_home)) > float(np.mean(empty_home)) + 0.5

    def test_coarse_fraction_maps_to_remaining_innings(self) -> None:
        sim = make_sim("MLB", live_context("MLB", 2, 1, 0.5))
        assert isinstance(sim, BaseballSimulator)
        assert sim._live_plan is not None
        assert sim._live_plan.start_inning == 6  # round(9 * 0.5) = 4 innings remain
        assert sim._live_plan.last_scheduled_inning == 9
        assert sim._live_plan.partial_cdf is None

        late = make_sim("MLB", live_context("MLB", 2, 1, 0.12))
        assert isinstance(late, BaseballSimulator)
        assert late._live_plan is not None
        assert late._live_plan.start_inning == 9

    def test_mid_extras_resume(self) -> None:
        context = live_context("MLB", 4, 4, 0.01, period=10, half="TOP", outs=1, bases="1--")
        home, away = simulate("MLB", context)
        assert not bool(np.any(home == away))
        assert int(home.min()) >= 4 and int(away.min()) >= 4
