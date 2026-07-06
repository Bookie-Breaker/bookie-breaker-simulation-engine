"""Hockey plugin tests: goal-rate adjustments, grid sampling, OT/SO resolution.

Regulation scores come off an exact 10x10 Dixon-Coles grid, so regulation
assertions compare seeded empirical frequencies against the analytic PMF
within 4-sigma binomial bands (the soccer pattern); OT/SO assertions use the
simulator's regulation-tie diagnostics so they are exact invariants.
"""

import time

import numpy as np
import pytest

from simulation_engine.clients.statistics import (
    AdvancedStats,
    DefensiveStats,
    HockeyStats,
    OffensiveStats,
    TeamStats,
)
from simulation_engine.core import league_averages as lg
from simulation_engine.core.params import GameContext
from simulation_engine.core.plugins import HOCKEY_GRID_CONFIG, get_plugin, get_simulator
from simulation_engine.core.plugins.hockey import (
    HockeyParams,
    HockeySimulator,
    map_hockey_stats,
)
from simulation_engine.core.poisson_grid import build_goal_grid
from simulation_engine.core.runner import GridConfig, run_monte_carlo

NHL_HOME = GameContext(league="NHL")
NHL_NEUTRAL = GameContext(league="NHL", neutral_site=True)
N = 200_000


def make_params(
    gf: float = lg.NHL_GOALS_PER_TEAM,
    ga: float = lg.NHL_GOALS_PER_TEAM,
    pp: float = lg.NHL_LEAGUE_PP_PCT,
    pk: float = lg.NHL_LEAGUE_PK_PCT,
    save_pct: float = lg.NHL_LEAGUE_SAVE_PCT,
) -> HockeyParams:
    return HockeyParams(
        goals_for_per_game=gf,
        goals_against_per_game=ga,
        power_play_pct=pp,
        penalty_kill_pct=pk,
        team_save_pct=save_pct,
    )


def make_simulator(
    home: HockeyParams | None = None,
    away: HockeyParams | None = None,
    context: GameContext = NHL_HOME,
) -> HockeySimulator:
    spec = get_plugin("NHL")
    sim = spec.simulator(dict(spec.plugin_config))
    assert isinstance(sim, HockeySimulator)
    sim.set_parameters(home or make_params(), away or make_params(), context)
    return sim


def binomial_4sigma(p: float, n: int) -> float:
    return 4.0 * float(np.sqrt(p * (1.0 - p) / n))


class TestGoalRates:
    def test_even_matchup_applies_home_multiplier(self) -> None:
        sim = make_simulator()
        assert sim._lam_home == pytest.approx(lg.NHL_GOALS_PER_TEAM * lg.NHL_HOME_GOAL_MULT)
        assert sim._lam_away == pytest.approx(lg.NHL_GOALS_PER_TEAM)

    def test_neutral_site_drops_home_multiplier(self) -> None:
        sim = make_simulator(context=NHL_NEUTRAL)
        assert sim._lam_home == pytest.approx(lg.NHL_GOALS_PER_TEAM)
        assert sim._lam_away == pytest.approx(lg.NHL_GOALS_PER_TEAM)

    def test_multiplicative_strength_blend(self) -> None:
        # Home scores 3.6 GF/G against a defense allowing 3.3 GA/G:
        # 3.6 * 3.3 / 3.0 * 1.05, with league-average special teams (mult 1).
        sim = make_simulator(home=make_params(gf=3.6), away=make_params(ga=3.3))
        assert sim._lam_home == pytest.approx(3.6 * 3.3 / 3.0 * lg.NHL_HOME_GOAL_MULT)
        assert sim._lam_away == pytest.approx(lg.NHL_GOALS_PER_TEAM)

    def test_strong_power_play_raises_own_rate(self) -> None:
        boosted = make_simulator(home=make_params(pp=0.27))
        base = make_simulator()
        assert boosted._lam_home == pytest.approx(base._lam_home * (1.0 + lg.HOCKEY_PP_WEIGHT * 0.06))
        assert boosted._lam_away == base._lam_away  # opponent's PP does not enter this side

    def test_strong_opposing_penalty_kill_lowers_own_rate(self) -> None:
        suppressed = make_simulator(away=make_params(pk=0.85))
        base = make_simulator()
        assert suppressed._lam_home == pytest.approx(base._lam_home * (1.0 - lg.HOCKEY_PK_WEIGHT * 0.06))
        assert suppressed._lam_away == base._lam_away

    def test_lambda_clamped_to_bounds(self) -> None:
        sim = make_simulator(home=make_params(gf=9.0, ga=0.1), away=make_params(gf=0.1, ga=9.0))
        assert sim._lam_home == 6.0  # 9.0 * 9.0 / 3.0 clamped down
        assert sim._lam_away == 1.0  # 0.1 * 0.1 / 3.0 clamped up

    def test_requires_hockey_params(self, make_team_params) -> None:
        sim = HockeySimulator({})
        with pytest.raises(TypeError, match="HockeyParams"):
            sim.set_parameters(make_team_params("h"), make_team_params("a"), NHL_HOME)

    def test_simulating_before_set_parameters_raises(self) -> None:
        with pytest.raises(RuntimeError, match="set_parameters"):
            HockeySimulator({}).simulate_games(np.random.default_rng(1), 10)


class TestRegulationGrid:
    def test_grid_matches_shared_builder(self) -> None:
        sim = make_simulator()
        assert sim._grid is not None
        expected = build_goal_grid(sim._lam_home, sim._lam_away, lg.HOCKEY_DC_RHO, 10)
        assert np.array_equal(sim._grid, expected)
        assert sim._grid.shape == (10, 10)
        assert sim._grid.sum() == pytest.approx(1.0)

    def test_empirical_regulation_frequencies_match_exact_grid_pmf(self) -> None:
        sim = make_simulator()
        assert sim._grid is not None
        home, away = sim._sample_regulation(np.random.default_rng(7), N)
        counts = np.zeros((10, 10))
        np.add.at(counts, (home, away), 1.0)
        empirical = counts / N
        for h in range(10):
            for a in range(10):
                p = float(sim._grid[h, a])
                if p >= 1e-4:
                    assert abs(empirical[h, a] - p) <= binomial_4sigma(p, N), f"cell ({h},{a})"

    def test_regulation_tie_rate_matches_grid_diagonal(self) -> None:
        sim = make_simulator()
        assert sim._grid is not None
        p_tie = float(np.trace(sim._grid))
        sim.simulate_games(np.random.default_rng(13), N)
        empirical = float(sim._last_regulation_tied.mean())
        assert abs(empirical - p_tie) <= binomial_4sigma(p_tie, N)
        assert 0.10 < p_tie < 0.30  # plausible NHL regulation-tie rate


class TestOvertimeShootout:
    def test_no_draws_in_output(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(19), N)
        assert int(np.sum(home == away)) == 0
        assert home.dtype == np.int32 and away.dtype == np.int32
        assert home.min() >= 0 and away.min() >= 0

    def test_ot_decided_games_have_score_differential_exactly_one(self) -> None:
        """NHL convention: OT and shootout wins add exactly one goal to the final."""
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(23), N)
        tied = sim._last_regulation_tied
        assert tied.any()
        assert (np.abs(home[tied].astype(np.int64) - away[tied].astype(np.int64)) == 1).all()

    def test_roughly_half_of_ties_decided_in_overtime(self) -> None:
        sim = make_simulator()
        sim.simulate_games(np.random.default_rng(29), N)
        total_ties = sim._last_ot_decided + sim._last_so_decided
        assert total_ties == int(sim._last_regulation_tied.sum())
        ot_fraction = sim._last_ot_decided / total_ties
        assert ot_fraction == pytest.approx(lg.NHL_OT_SHARE_OF_TIES, abs=0.02)

    def test_tie_winner_weighted_by_relative_goal_rate(self) -> None:
        sim = make_simulator(home=make_params(gf=3.9), away=make_params(gf=2.4))
        home, away = sim.simulate_games(np.random.default_rng(31), N)
        tied = sim._last_regulation_tied
        p_home = sim._lam_home / (sim._lam_home + sim._lam_away)
        home_win_rate = float(np.mean(home[tied] > away[tied]))
        assert p_home > 0.6
        assert home_win_rate == pytest.approx(p_home, abs=binomial_4sigma(p_home, int(tied.sum())))

    def test_even_matchup_is_near_coin_flip_at_neutral_site(self) -> None:
        sim = make_simulator(context=NHL_NEUTRAL)
        home, away = sim.simulate_games(np.random.default_rng(37), N)
        assert float(np.mean(home > away)) == pytest.approx(0.5, abs=0.01)


class TestSampling:
    def test_nhl_totals_in_plausible_range(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(41), 50_000)
        total_mean = float((home + away).mean())
        assert 5.5 < total_mean < 7.0  # ~6.15 regulation plus the OT/SO winner goal
        assert home.max() <= 10 and away.max() <= 10  # 0-9 grid plus at most one OT goal

    def test_single_game_contract(self) -> None:
        result = make_simulator().simulate_game(np.random.default_rng(5))
        assert result.home_score >= 0
        assert result.away_score >= 0
        assert result.home_score != result.away_score

    def test_same_seed_is_deterministic(self) -> None:
        sim = make_simulator()
        a = sim.simulate_games(np.random.default_rng(42), 10_000)
        b = sim.simulate_games(np.random.default_rng(42), 10_000)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_speed_sanity(self) -> None:
        sim = make_simulator()
        rng = np.random.default_rng(1)
        started = time.perf_counter()
        sim.simulate_games(rng, 10_000)
        assert time.perf_counter() - started < 1.0  # one grid draw plus a Bernoulli pass


class TestRunnerIntegration:
    def run(self, seed: int = 11):
        sim = get_simulator("NHL")
        return run_monte_carlo(
            sim,
            make_params(gf=3.4, ga=2.7),
            make_params(gf=2.9, ga=3.1),
            NHL_HOME,
            iterations=20_000,
            convergence_threshold=1e-9,
            seed=seed,
            grid_config=HOCKEY_GRID_CONFIG,
        )

    def test_no_draw_probability_and_partition(self) -> None:
        out = self.run()
        assert out.draw_prob == 0.0
        assert out.home_win_prob + out.away_win_prob == pytest.approx(1.0)

    def test_hockey_grid_sizes(self) -> None:
        out = self.run()
        assert len(out.spread_covers) == 2 * HOCKEY_GRID_CONFIG.spread_radius + 2
        assert len(out.total_overs) == 2 * HOCKEY_GRID_CONFIG.total_radius + 2
        assert all(not float(line).is_integer() for line in out.spread_covers)

    def test_same_seed_identical_runs(self) -> None:
        a = self.run(seed=99)
        b = self.run(seed=99)
        assert np.array_equal(a.home_scores, b.home_scores)
        assert np.array_equal(a.away_scores, b.away_scores)


class TestMapHockeyStats:
    def make_stats(self, hockey: HockeyStats) -> TeamStats:
        return TeamStats(
            team_id="t-hky",
            team_abbreviation="COL",
            offensive=OffensiveStats(),
            defensive=DefensiveStats(),
            advanced=AdvancedStats(),
            hockey=hockey,
        )

    def test_populated_block_maps_directly(self) -> None:
        params = map_hockey_stats(
            self.make_stats(
                HockeyStats(
                    goals_for_per_game=3.5,
                    goals_against_per_game=2.6,
                    shots_for_per_game=32.1,
                    shots_against_per_game=27.8,
                    power_play_pct=0.245,
                    penalty_kill_pct=0.815,
                    team_save_pct=0.912,
                )
            )
        )
        assert params == HockeyParams(
            goals_for_per_game=3.5,
            goals_against_per_game=2.6,
            power_play_pct=0.245,
            penalty_kill_pct=0.815,
            team_save_pct=0.912,
        )

    def test_empty_block_falls_back_to_nhl_league_averages(self) -> None:
        params = map_hockey_stats(self.make_stats(HockeyStats()))
        assert params.goals_for_per_game == lg.NHL_GOALS_PER_TEAM
        assert params.goals_against_per_game == lg.NHL_GOALS_PER_TEAM
        assert params.power_play_pct == lg.NHL_LEAGUE_PP_PCT
        assert params.penalty_kill_pct == lg.NHL_LEAGUE_PK_PCT
        assert params.team_save_pct == lg.NHL_LEAGUE_SAVE_PCT


class TestRegistry:
    def test_nhl_spec(self) -> None:
        spec = get_plugin("nhl")
        assert spec.label == "hockey"
        assert spec.simulator is HockeySimulator
        assert spec.map_team_stats is map_hockey_stats
        assert spec.grid_config == GridConfig(spread_radius=3, total_radius=4)
        assert spec.plugin_config == {
            "league_goals_per_team": 3.0,
            "home_goal_mult": 1.05,
            "pp_weight": 0.5,
            "pk_weight": 0.5,
            "league_pp_pct": 0.21,
            "league_pk_pct": 0.79,
            "dc_rho": -0.05,
        }

    def test_ncaa_hky_stays_gated(self) -> None:
        from simulation_engine.api.errors import UnprocessableError

        with pytest.raises(UnprocessableError, match="not supported"):
            get_plugin("NCAA_HKY")  # gated per ADR-026

    def test_sport_and_league_identity(self) -> None:
        sim = get_simulator("NHL")
        assert sim.get_sport() == "HOCKEY"
        assert sim.get_league() == "NHL"
        assert isinstance(sim, HockeySimulator)
