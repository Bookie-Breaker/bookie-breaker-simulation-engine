"""Soccer plugin tests: exact-PMF statistical assertions, not loose tolerances.

The Dixon-Coles grid is an exact discrete distribution, so sampling tests
compare seeded empirical frequencies against the analytic grid PMF within a
4-sigma binomial standard error — deterministic given the seed, and tight.
"""

import time

import numpy as np
import pytest

from simulation_engine.clients.statistics import (
    AdvancedStats,
    DefensiveStats,
    OffensiveStats,
    SoccerStats,
    TeamStats,
)
from simulation_engine.core import league_averages as lg
from simulation_engine.core.params import GameContext
from simulation_engine.core.plugins import SOCCER_GRID_CONFIG, get_plugin, get_simulator
from simulation_engine.core.plugins.soccer import (
    SoccerParams,
    SoccerSimulator,
    _build_goal_grid,
    _poisson_pmf,
    map_soccer_stats,
)
from simulation_engine.core.runner import GridConfig, run_monte_carlo

WC_NEUTRAL = GameContext(league="FIFA_WC", neutral_site=True)
EPL_HOME = GameContext(league="EPL")
N = 200_000

TAU_CELLS = ((0, 0), (1, 0), (0, 1), (1, 1))


def make_params(attack: float = 1.0, defense: float = 1.0) -> SoccerParams:
    return SoccerParams(
        attack=attack,
        defense=defense,
        goals_for_per_match=attack * lg.SOCCER_WC_BASE_GOALS_PER_TEAM,
        goals_against_per_match=defense * lg.SOCCER_WC_BASE_GOALS_PER_TEAM,
    )


def make_simulator(context: GameContext = WC_NEUTRAL, league: str = "FIFA_WC", **params: float) -> SoccerSimulator:
    home = make_params(params.get("home_attack", 1.15), params.get("home_defense", 0.9))
    away = make_params(params.get("away_attack", 0.95), params.get("away_defense", 1.05))
    sim = get_plugin(league).simulator(dict(get_plugin(league).plugin_config))
    assert isinstance(sim, SoccerSimulator)
    sim.set_parameters(home, away, context)
    return sim


def margin_pmf(grid: np.ndarray) -> dict[int, float]:
    """Analytic P(margin == m) from the grid (Skellam-like, truncated)."""
    return {m: float(np.trace(grid, offset=-m)) for m in range(-12, 13)}


def total_pmf(grid: np.ndarray) -> dict[int, float]:
    """Analytic P(total == t) from the grid."""
    flipped = np.flipud(grid)  # flipped[i, i+k] = grid[12-i, i+k], a total of 12+k
    return {t: float(np.trace(flipped, offset=t - 12)) for t in range(0, 25)}


def binomial_4sigma(p: float, n: int) -> float:
    return 4.0 * float(np.sqrt(p * (1.0 - p) / n))


class TestGoalGrid:
    def test_poisson_pmf_matches_closed_form(self) -> None:
        lam = 1.7
        pmf = _poisson_pmf(lam)
        factorials = [float(np.prod(np.arange(1, k + 1))) if k else 1.0 for k in range(13)]
        expected = [np.exp(-lam) * lam**k / factorials[k] for k in range(13)]
        assert pmf == pytest.approx(expected)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-4)  # >0.9999 of the mass at soccer rates

    def test_grid_sums_to_one_and_is_nonnegative(self) -> None:
        grid = _build_goal_grid(1.5, 1.1, lg.SOCCER_DC_RHO)
        assert grid.shape == (13, 13)
        assert grid.sum() == pytest.approx(1.0)
        assert (grid >= 0).all()

    def test_tau_adjusts_exactly_the_four_low_score_cells(self) -> None:
        lam, mu, rho = 1.6, 1.2, lg.SOCCER_DC_RHO
        adjusted = _build_goal_grid(lam, mu, rho)
        plain = _build_goal_grid(lam, mu, 0.0)

        # Outside the tau cells the two grids differ only by renormalization.
        ratio = adjusted / plain
        mask = np.ones((13, 13), dtype=bool)
        for cell in TAU_CELLS:
            mask[cell] = False
        scale = float(ratio[mask][0])
        assert ratio[mask] == pytest.approx(scale)

        # The tau cells carry exactly the specified corrections on top.
        assert adjusted[0, 0] / plain[0, 0] == pytest.approx(scale * (1.0 - lam * mu * rho))
        assert adjusted[1, 0] / plain[1, 0] == pytest.approx(scale * (1.0 + mu * rho))
        assert adjusted[0, 1] / plain[0, 1] == pytest.approx(scale * (1.0 + lam * rho))
        assert adjusted[1, 1] / plain[1, 1] == pytest.approx(scale * (1.0 - rho))

    def test_negative_rho_boosts_draws(self) -> None:
        adjusted = _build_goal_grid(1.4, 1.4, lg.SOCCER_DC_RHO)
        plain = _build_goal_grid(1.4, 1.4, 0.0)
        assert float(np.trace(adjusted)) > float(np.trace(plain))

    def test_extreme_rho_clamps_negative_cells_and_renormalizes(self) -> None:
        # tau(0,0) = 1 - 2.0*2.0*0.9 = -2.6 -> clamped to zero
        grid = _build_goal_grid(2.0, 2.0, 0.9)
        assert grid[0, 0] == 0.0
        assert (grid >= 0).all()
        assert grid.sum() == pytest.approx(1.0)


class TestLambdaAndContext:
    def test_neutral_site_forces_home_multiplier_to_one(self) -> None:
        neutral = make_simulator(GameContext(league="EPL", neutral_site=True), league="EPL")
        home = make_simulator(EPL_HOME, league="EPL")
        assert neutral._lam_away == home._lam_away
        assert home._lam_home == pytest.approx(neutral._lam_home * lg.SOCCER_EPL_HOME_GOAL_MULTIPLIER)

    def test_epl_config_applies_base_rate_and_multiplier(self) -> None:
        sim = get_plugin("EPL").simulator(dict(get_plugin("EPL").plugin_config))
        sim.set_parameters(make_params(), make_params(), EPL_HOME)
        assert isinstance(sim, SoccerSimulator)
        assert sim._lam_home == pytest.approx(lg.SOCCER_EPL_BASE_GOALS_PER_TEAM * lg.SOCCER_EPL_HOME_GOAL_MULTIPLIER)
        assert sim._lam_away == pytest.approx(lg.SOCCER_EPL_BASE_GOALS_PER_TEAM)

    def test_fifa_wc_config_has_no_home_boost(self) -> None:
        sim = get_plugin("FIFA_WC").simulator(dict(get_plugin("FIFA_WC").plugin_config))
        sim.set_parameters(make_params(), make_params(), GameContext(league="FIFA_WC"))
        assert isinstance(sim, SoccerSimulator)
        assert sim._lam_home == pytest.approx(lg.SOCCER_WC_BASE_GOALS_PER_TEAM)
        assert sim._lam_away == pytest.approx(lg.SOCCER_WC_BASE_GOALS_PER_TEAM)

    def test_lambda_clamped_to_bounds(self) -> None:
        sim = SoccerSimulator({})
        sim.set_parameters(make_params(attack=4.0, defense=4.0), make_params(attack=0.01, defense=4.0), WC_NEUTRAL)
        assert sim._lam_home == 4.5  # 1.35 * 4.0 * 4.0 clamped down
        assert sim._lam_away == 0.2  # 1.35 * 0.01 * 4.0 clamped up

    def test_requires_soccer_params(self, make_team_params) -> None:
        sim = SoccerSimulator({})
        with pytest.raises(TypeError, match="SoccerParams"):
            sim.set_parameters(make_team_params("h"), make_team_params("a"), WC_NEUTRAL)

    def test_simulating_before_set_parameters_raises(self) -> None:
        with pytest.raises(RuntimeError, match="set_parameters"):
            SoccerSimulator({}).simulate_games(np.random.default_rng(1), 10)


class TestSampling:
    def test_empirical_cell_frequencies_match_exact_grid_pmf(self) -> None:
        sim = make_simulator()
        assert sim._grid is not None
        home, away = sim.simulate_games(np.random.default_rng(7), N)
        counts = np.zeros((13, 13))
        np.add.at(counts, (home, away), 1.0)
        empirical = counts / N
        for h in range(13):
            for a in range(13):
                p = float(sim._grid[h, a])
                if p >= 1e-4:
                    assert abs(empirical[h, a] - p) <= binomial_4sigma(p, N), f"cell ({h},{a})"

    def test_empirical_draw_rate_matches_diagonal_sum(self) -> None:
        sim = make_simulator()
        assert sim._grid is not None
        p_draw = float(np.trace(sim._grid))
        home, away = sim.simulate_games(np.random.default_rng(13), N)
        assert abs(float(np.mean(home == away)) - p_draw) <= binomial_4sigma(p_draw, N)
        assert 0.15 < p_draw < 0.40  # plausible soccer draw rate

    def test_empirical_margin_distribution_matches_analytic(self) -> None:
        sim = make_simulator()
        assert sim._grid is not None
        analytic = margin_pmf(sim._grid)
        assert sum(analytic.values()) == pytest.approx(1.0)
        home, away = sim.simulate_games(np.random.default_rng(29), N)
        margins = home - away
        for m, p in analytic.items():
            if p >= 1e-4:
                assert abs(float(np.mean(margins == m)) - p) <= binomial_4sigma(p, N), f"margin {m}"

    def test_draws_are_valid_outcomes_and_scores_stay_in_grid(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(3), 20_000)
        assert int(np.sum(home == away)) > 0  # no overtime/tie-break (ADR-027)
        assert home.min() >= 0 and home.max() <= 12
        assert away.min() >= 0 and away.max() <= 12
        assert home.dtype == np.int32 and away.dtype == np.int32

    def test_single_game_contract(self) -> None:
        sim = make_simulator()
        result = sim.simulate_game(np.random.default_rng(5))
        assert 0 <= result.home_score <= 12
        assert 0 <= result.away_score <= 12

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
        assert time.perf_counter() - started < 1.0  # 10k games: single vectorized draw


class TestRunnerIntegration:
    def run(self, seed: int = 11, **kwargs):
        sim = get_simulator("FIFA_WC")
        output = run_monte_carlo(
            sim,
            make_params(1.15, 0.9),
            make_params(0.95, 1.05),
            WC_NEUTRAL,
            iterations=20_000,
            convergence_threshold=1e-9,
            seed=seed,
            grid_config=SOCCER_GRID_CONFIG,
            **kwargs,
        )
        assert isinstance(sim, SoccerSimulator)
        assert sim._grid is not None
        return output, sim._grid

    def test_draw_prob_flows_to_output_and_probabilities_partition(self) -> None:
        out, grid = self.run()
        assert out.draw_prob > 0.0
        assert out.home_win_prob + out.away_win_prob + out.draw_prob == pytest.approx(1.0)
        p_draw = float(np.trace(grid))
        assert abs(out.draw_prob - p_draw) <= binomial_4sigma(p_draw, out.iterations_run)

    def test_default_soccer_grids_are_half_point_lines(self) -> None:
        out, _ = self.run()
        assert len(out.spread_covers) == 2 * SOCCER_GRID_CONFIG.spread_radius + 2
        assert len(out.total_overs) == 2 * SOCCER_GRID_CONFIG.total_radius + 2
        assert all(not float(line).is_integer() for line in out.spread_covers)
        assert out.spread_pushes == {}
        assert out.total_pushes == {}

    def test_integer_lines_expose_analytic_push_probabilities(self) -> None:
        out, grid = self.run(common_spreads=[-1.0, 0.5], common_totals=[2.0, 2.5])
        n = out.iterations_run
        assert set(out.spread_pushes) == {-1.0}
        assert set(out.total_pushes) == {2.0}

        # Pushes equal the empirical event frequency exactly...
        assert out.spread_pushes[-1.0] == pytest.approx(float(np.mean(out.margins == 1)))
        assert out.total_pushes[2.0] == pytest.approx(float(np.mean(out.totals == 2)))
        # ...and the analytic grid probability within binomial standard error.
        p_margin_1 = margin_pmf(grid)[1]
        p_total_2 = total_pmf(grid)[2]
        assert abs(out.spread_pushes[-1.0] - p_margin_1) <= binomial_4sigma(p_margin_1, n)
        assert abs(out.total_pushes[2.0] - p_total_2) <= binomial_4sigma(p_total_2, n)

        # Cover/push/loss partition to 1 on integer lines.
        loss = float(np.mean(out.margins < 1))
        assert out.spread_covers[-1.0] + out.spread_pushes[-1.0] + loss == pytest.approx(1.0)

    def test_same_seed_identical_runs(self) -> None:
        a, _ = self.run(seed=99)
        b, _ = self.run(seed=99)
        assert np.array_equal(a.home_scores, b.home_scores)
        assert np.array_equal(a.away_scores, b.away_scores)
        assert a.draw_prob == b.draw_prob


class TestMapSoccerStats:
    def make_stats(self, soccer: SoccerStats) -> TeamStats:
        return TeamStats(
            team_id="t-soc",
            team_abbreviation="BRA",
            offensive=OffensiveStats(),
            defensive=DefensiveStats(),
            advanced=AdvancedStats(),
            soccer=soccer,
        )

    def test_populated_block_maps_directly(self) -> None:
        params = map_soccer_stats(
            self.make_stats(
                SoccerStats(
                    goals_for_per_match=2.1,
                    goals_against_per_match=0.8,
                    attack_strength=1.4,
                    defense_strength=0.7,
                    draws=3,
                    form_goals_for_last5=11.0,
                    form_goals_against_last5=4.0,
                    form_points_last5=13,
                )
            )
        )
        assert params == SoccerParams(attack=1.4, defense=0.7, goals_for_per_match=2.1, goals_against_per_match=0.8)

    def test_empty_block_falls_back_to_league_average_strengths(self) -> None:
        params = map_soccer_stats(self.make_stats(SoccerStats()))
        assert params.attack == 1.0
        assert params.defense == 1.0
        assert params.goals_for_per_match == 0.0
        assert params.goals_against_per_match == 0.0


class TestRegistry:
    def test_fifa_wc_spec(self) -> None:
        spec = get_plugin("fifa_wc")
        assert spec.label == "soccer"
        assert spec.simulator is SoccerSimulator
        assert spec.map_team_stats is map_soccer_stats
        assert spec.grid_config == GridConfig(spread_radius=3, total_radius=4)
        assert spec.plugin_config == {
            "base_goals_per_team": 1.35,
            "home_goal_multiplier": 1.0,
            "dc_rho": -0.11,
        }

    def test_epl_spec(self) -> None:
        spec = get_plugin("EPL")
        assert spec.label == "soccer"
        assert spec.grid_config == SOCCER_GRID_CONFIG
        assert spec.plugin_config == {
            "base_goals_per_team": 1.45,
            "home_goal_multiplier": 1.15,
            "dc_rho": -0.11,
        }

    def test_sport_and_league_identity(self) -> None:
        sim = get_simulator("FIFA_WC")
        assert sim.get_sport() == "SOCCER"
        assert sim.get_league() == "FIFA_WC"
        assert isinstance(sim, SoccerSimulator)
        sim.set_parameters(make_params(), make_params(), EPL_HOME)
        assert sim.get_league() == "EPL"  # adopted from the game context

    def test_league_config_override_wins(self) -> None:
        sim = SoccerSimulator({"league": "EPL"})
        sim.set_parameters(make_params(), make_params(), GameContext(league="FIFA_WC"))
        assert sim.get_league() == "EPL"
