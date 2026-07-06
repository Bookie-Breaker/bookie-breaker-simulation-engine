"""Baseball plugin tests: calibration exactness, starter multipliers, game flow.

The half-inning PMFs are exact discrete distributions, so distribution-level
assertions are analytic (1e-6 mean matching, P(0) band, over-dispersion);
game-flow assertions (bottom-9 skip, extra innings) use seeded large samples
against analytic expectations within 4-sigma bands.
"""

import time

import numpy as np
import pytest

from simulation_engine.clients.statistics import (
    AdvancedStats,
    BaseballStats,
    DefensiveStats,
    OffensiveStats,
    TeamStats,
)
from simulation_engine.core import league_averages as lg
from simulation_engine.core.params import GameContext
from simulation_engine.core.plugins import BASEBALL_GRID_CONFIG, get_plugin, get_simulator
from simulation_engine.core.plugins.baseball import (
    BaseballParams,
    BaseballSimulator,
    _half_inning_pmf,
    _zero_modified_geometric_pmf,
    map_baseball_stats,
)
from simulation_engine.core.runner import GridConfig, run_monte_carlo

MLB = GameContext(league="MLB")
RUNS = np.arange(11)
N = 200_000
MLB_HALF_INNING_MEAN = lg.MLB_RUNS_PER_GAME / 9.0


def make_params(
    rs: float = 4.5, ra: float = 4.5, era: float = 4.2, fip: float = 4.1, bullpen: float = 4.2
) -> BaseballParams:
    return BaseballParams(
        runs_scored_per_game=rs, runs_allowed_per_game=ra, team_era=era, team_fip=fip, bullpen_era=bullpen
    )


def make_simulator(
    home: BaseballParams | None = None,
    away: BaseballParams | None = None,
    context: GameContext = MLB,
    league: str = "MLB",
) -> BaseballSimulator:
    spec = get_plugin(league)
    sim = spec.simulator(dict(spec.plugin_config))
    assert isinstance(sim, BaseballSimulator)
    sim.set_parameters(home or make_params(), away or make_params(), context)
    return sim


def pmf_mean(pmf: np.ndarray) -> float:
    return float(np.dot(pmf, RUNS))


def four_sigma(std: float, n: int) -> float:
    return 4.0 * std / float(np.sqrt(n))


class TestHalfInningDistribution:
    def test_pmf_shape_and_zero_modification(self) -> None:
        pmf = _zero_modified_geometric_pmf(0.73, 0.4)
        assert pmf.shape == (11,)
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= 0).all()
        assert pmf[0] == 0.73  # truncation renormalizes the tail, not P(0)
        # Geometric tail: constant ratio between successive k >= 1 masses.
        ratios = pmf[2:] / pmf[1:-1]
        assert ratios == pytest.approx([0.4] * 9)

    @pytest.mark.parametrize("league_mean", [MLB_HALF_INNING_MEAN, lg.NCAA_BSB_RUNS_PER_GAME / 9.0])
    def test_calibrated_mean_matches_target_to_1e6_across_range(self, league_mean: float) -> None:
        for target in np.linspace(0.2, 1.2, 51):
            pmf = _half_inning_pmf(float(target), league_mean)
            assert abs(pmf_mean(pmf) - float(target)) < 1e-6, f"target {target}"

    @pytest.mark.parametrize("league_mean", [MLB_HALF_INNING_MEAN, lg.NCAA_BSB_RUNS_PER_GAME / 9.0])
    def test_zero_probability_stays_in_realistic_band(self, league_mean: float) -> None:
        for target in np.linspace(0.2, 1.2, 51):
            pmf = _half_inning_pmf(float(target), league_mean)
            assert 0.55 <= pmf[0] <= 0.85, f"target {target}: P(0)={pmf[0]}"

    @pytest.mark.parametrize("league_mean", [MLB_HALF_INNING_MEAN, lg.NCAA_BSB_RUNS_PER_GAME / 9.0])
    def test_over_dispersed_like_real_innings(self, league_mean: float) -> None:
        for target in np.linspace(0.2, 1.2, 51):
            pmf = _half_inning_pmf(float(target), league_mean)
            mean = pmf_mean(pmf)
            variance = float(np.dot(pmf, (RUNS - mean) ** 2))
            assert variance / mean > 1.0, f"target {target}: VMR={variance / mean}"

    def test_league_average_target_uses_p0_base(self) -> None:
        pmf = _half_inning_pmf(MLB_HALF_INNING_MEAN, MLB_HALF_INNING_MEAN)
        assert pmf[0] == pytest.approx(lg.BASEBALL_P0_BASE)

    def test_target_clamped_to_calibratable_range(self) -> None:
        low = _half_inning_pmf(0.05, MLB_HALF_INNING_MEAN)
        high = _half_inning_pmf(3.0, MLB_HALF_INNING_MEAN)
        assert pmf_mean(low) == pytest.approx(0.2, abs=1e-6)
        assert pmf_mean(high) == pytest.approx(1.2, abs=1e-6)


class TestExpectedRuns:
    def test_even_league_average_matchup_hits_league_rate(self) -> None:
        sim = make_simulator()
        models = sim._models()
        for model in models:
            # Bullpen ERA 4.2 == league ERA -> multiplier 1.0; no starter -> 1.0.
            assert pmf_mean(model.starter_pmf) == pytest.approx(MLB_HALF_INNING_MEAN, abs=1e-6)
            assert pmf_mean(model.bullpen_pmf) == pytest.approx(MLB_HALF_INNING_MEAN, abs=1e-6)

    def test_odds_ratio_blend(self) -> None:
        # home offense 5.4 RS/G vs away 5.0 RA/G: 5.4 * 5.0 / 4.5 / 9 = 2/3.
        sim = make_simulator(home=make_params(rs=5.4), away=make_params(ra=5.0))
        home_batting, away_batting = sim._models()
        assert pmf_mean(home_batting.starter_pmf) == pytest.approx(5.4 * 5.0 / 4.5 / 9.0, abs=1e-6)
        assert pmf_mean(away_batting.starter_pmf) == pytest.approx(MLB_HALF_INNING_MEAN, abs=1e-6)

    def test_requires_baseball_params(self, make_team_params) -> None:
        sim = BaseballSimulator({})
        with pytest.raises(TypeError, match="BaseballParams"):
            sim.set_parameters(make_team_params("h"), make_team_params("a"), MLB)

    def test_simulating_before_set_parameters_raises(self) -> None:
        with pytest.raises(RuntimeError, match="set_parameters"):
            BaseballSimulator({}).simulate_games(np.random.default_rng(1), 10)


class TestStarterMultiplier:
    def test_away_starter_scales_home_batting_innings_1_to_6_only(self) -> None:
        sim = make_simulator(context=GameContext(league="MLB", away_starter_fip=6.0))
        neutral = make_simulator()
        home_batting, away_batting = sim._models()
        expected = MLB_HALF_INNING_MEAN * (6.0 / lg.MLB_LEAGUE_FIP)
        assert pmf_mean(home_batting.starter_pmf) == pytest.approx(expected, abs=1e-6)
        # Bullpen phase (innings 7+) is untouched by the starter.
        assert np.array_equal(home_batting.bullpen_pmf, neutral._models()[0].bullpen_pmf)
        # The AWAY batting side never sees the away starter.
        assert np.array_equal(away_batting.starter_pmf, neutral._models()[1].starter_pmf)

    def test_home_starter_scales_away_batting(self) -> None:
        sim = make_simulator(context=GameContext(league="MLB", home_starter_fip=2.5))
        neutral = make_simulator()
        home_batting, away_batting = sim._models()
        expected = MLB_HALF_INNING_MEAN * (2.5 / lg.MLB_LEAGUE_FIP)
        assert pmf_mean(away_batting.starter_pmf) == pytest.approx(expected, abs=1e-6)
        assert np.array_equal(home_batting.starter_pmf, neutral._models()[0].starter_pmf)

    def test_multiplier_clipped(self) -> None:
        bad = make_simulator(context=GameContext(league="MLB", away_starter_fip=12.0))
        good = make_simulator(context=GameContext(league="MLB", away_starter_fip=0.5))
        assert pmf_mean(bad._models()[0].starter_pmf) == pytest.approx(MLB_HALF_INNING_MEAN * 1.6, abs=1e-6)
        assert pmf_mean(good._models()[0].starter_pmf) == pytest.approx(MLB_HALF_INNING_MEAN * 0.6, abs=1e-6)

    def test_per_inning_run_means_shift_in_starter_innings_only(self) -> None:
        """Seeded sims: FIP 6.0 vs 2.5 away starters move HOME per-inning scoring
        by the analytic amount in innings 1-6 and not at all in innings 7-9."""
        bad = make_simulator(context=GameContext(league="MLB", away_starter_fip=6.0))
        good = make_simulator(context=GameContext(league="MLB", away_starter_fip=2.5))
        bad_home, _ = bad._simulate_regulation(np.random.default_rng(101), N)
        good_home, _ = good._simulate_regulation(np.random.default_rng(202), N)

        bad_mean = MLB_HALF_INNING_MEAN * (6.0 / lg.MLB_LEAGUE_FIP)
        good_mean = MLB_HALF_INNING_MEAN * (2.5 / lg.MLB_LEAGUE_FIP)
        tol = four_sigma(1.5, N)  # half-inning run std stays below ~1.4 at these rates
        for inning in range(6):
            assert float(bad_home[:, inning].mean()) == pytest.approx(bad_mean, abs=tol), f"inning {inning + 1}"
            assert float(good_home[:, inning].mean()) == pytest.approx(good_mean, abs=tol), f"inning {inning + 1}"
        # Innings 7-8 revert to the identical bullpen rate (9th has the skip).
        for inning in range(6, 8):
            assert float(bad_home[:, inning].mean()) == pytest.approx(MLB_HALF_INNING_MEAN, abs=tol)
            assert float(good_home[:, inning].mean()) == pytest.approx(MLB_HALF_INNING_MEAN, abs=tol)

    def test_bullpen_multiplier_falls_back_to_team_era_then_neutral(self) -> None:
        weak_pen = make_simulator(away=make_params(bullpen=5.5))
        assert pmf_mean(weak_pen._models()[0].bullpen_pmf) == pytest.approx(
            MLB_HALF_INNING_MEAN * (5.5 / lg.MLB_LEAGUE_ERA), abs=1e-6
        )
        via_team_era = make_simulator(away=make_params(era=5.0, bullpen=0.0))
        assert pmf_mean(via_team_era._models()[0].bullpen_pmf) == pytest.approx(
            MLB_HALF_INNING_MEAN * (5.0 / lg.MLB_LEAGUE_ERA), abs=1e-6
        )
        no_era = make_simulator(away=make_params(era=0.0, bullpen=0.0))
        assert pmf_mean(no_era._models()[0].bullpen_pmf) == pytest.approx(MLB_HALF_INNING_MEAN, abs=1e-6)


class TestBottomNineSkip:
    def test_home_never_scores_bottom_nine_when_leading_after_eight_and_a_half(self) -> None:
        sim = make_simulator()
        home_by_inning, away_by_inning = sim._simulate_regulation(np.random.default_rng(31), 50_000)
        home_leads = home_by_inning[:, :8].sum(axis=1) > away_by_inning.sum(axis=1)
        assert home_leads.any()
        assert (home_by_inning[home_leads, 8] == 0).all()
        # And games where home does NOT lead do sometimes score in the 9th.
        assert (home_by_inning[~home_leads, 8] > 0).any()

    def test_regulation_totals_mean_below_no_skip_control(self) -> None:
        sim = make_simulator()
        home_by_inning, away_by_inning = sim._simulate_regulation(np.random.default_rng(37), N)
        totals_mean = float((home_by_inning.sum(axis=1) + away_by_inning.sum(axis=1)).mean())
        # No-skip control: 18 half-innings at the league rate = 9.0 runs.
        no_skip_mean = 18.0 * MLB_HALF_INNING_MEAN
        reduction = no_skip_mean - totals_mean
        assert 0.1 < reduction < 0.45  # the documented ~0.2-0.4 run totals bias


class TestExtraInnings:
    def test_no_ties_in_output(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(7), N)
        assert int(np.sum(home == away)) == 0
        assert home.dtype == np.int32 and away.dtype == np.int32
        assert home.min() >= 0 and away.min() >= 0

    def test_extra_inning_frequency_in_plausible_band(self) -> None:
        sim = make_simulator()
        sim.simulate_games(np.random.default_rng(7), N)
        frequency = sim._last_extra_inning_games / N
        assert 0.06 <= frequency <= 0.14  # ~8-12% at even matchups, with margin

    def test_safety_cap_never_triggers_at_typical_parameters(self) -> None:
        sim = make_simulator()
        for seed in (1, 2, 3):
            sim.simulate_games(np.random.default_rng(seed), N)
            assert sim._last_forced_tiebreaks == 0

    def test_even_matchup_is_near_coin_flip(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(19), N)
        assert float(np.mean(home > away)) == pytest.approx(0.5, abs=0.01)


class TestSampling:
    def test_mlb_scores_in_plausible_range(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(11), 50_000)
        total_mean = float((home + away).mean())
        assert 8.0 < total_mean < 10.0  # ~9 minus the bottom-9 skip, plus extras
        assert home.max() < 60 and away.max() < 60

    def test_ncaa_bsb_scores_higher_than_mlb(self) -> None:
        ncaa_params = make_params(rs=6.5, ra=6.5)
        ncaa = make_simulator(ncaa_params, ncaa_params, GameContext(league="NCAA_BSB"), league="NCAA_BSB")
        mlb = make_simulator()
        ncaa_home, ncaa_away = ncaa.simulate_games(np.random.default_rng(23), 50_000)
        mlb_home, mlb_away = mlb.simulate_games(np.random.default_rng(23), 50_000)
        assert float((ncaa_home + ncaa_away).mean()) > float((mlb_home + mlb_away).mean()) + 2.0

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
        assert time.perf_counter() - started < 1.0  # same budget as the other plugins


class TestRunnerIntegration:
    def run(self, seed: int = 11, context: GameContext = MLB):
        sim = get_simulator("MLB")
        return run_monte_carlo(
            sim,
            make_params(rs=4.9),
            make_params(ra=4.2),
            context,
            iterations=20_000,
            convergence_threshold=1e-9,
            seed=seed,
            grid_config=BASEBALL_GRID_CONFIG,
        )

    def test_no_draw_probability_and_partition(self) -> None:
        out = self.run()
        assert out.draw_prob == 0.0
        assert out.home_win_prob + out.away_win_prob == pytest.approx(1.0)

    def test_baseball_grid_sizes(self) -> None:
        out = self.run()
        assert len(out.spread_covers) == 2 * BASEBALL_GRID_CONFIG.spread_radius + 2
        assert len(out.total_overs) == 2 * BASEBALL_GRID_CONFIG.total_radius + 2
        assert all(not float(line).is_integer() for line in out.spread_covers)

    def test_same_seed_identical_runs(self) -> None:
        a = self.run(seed=99)
        b = self.run(seed=99)
        assert np.array_equal(a.home_scores, b.home_scores)
        assert np.array_equal(a.away_scores, b.away_scores)

    def test_starter_context_shifts_run_output(self) -> None:
        neutral = self.run(seed=7)
        bad_away_starter = self.run(seed=7, context=GameContext(league="MLB", away_starter_fip=6.0))
        assert float(bad_away_starter.home_scores.mean()) > float(neutral.home_scores.mean()) + 0.5


class TestMapBaseballStats:
    def make_stats(self, baseball: BaseballStats) -> TeamStats:
        return TeamStats(
            team_id="t-bsb",
            team_abbreviation="NYY",
            offensive=OffensiveStats(),
            defensive=DefensiveStats(),
            advanced=AdvancedStats(),
            baseball=baseball,
        )

    def test_populated_block_maps_directly(self) -> None:
        params = map_baseball_stats(
            self.make_stats(
                BaseballStats(
                    runs_scored_per_game=5.2,
                    runs_allowed_per_game=3.9,
                    team_woba=0.330,
                    team_obp=0.335,
                    team_slg=0.442,
                    batting_strikeout_pct=21.5,
                    batting_walk_pct=8.9,
                    team_era=3.61,
                    team_fip=3.75,
                    bullpen_era=3.9,
                )
            )
        )
        assert params == BaseballParams(
            runs_scored_per_game=5.2,
            runs_allowed_per_game=3.9,
            team_era=3.61,
            team_fip=3.75,
            bullpen_era=3.9,
        )

    def test_empty_block_falls_back_to_mlb_league_averages(self) -> None:
        params = map_baseball_stats(self.make_stats(BaseballStats()))
        assert params.runs_scored_per_game == lg.MLB_RUNS_PER_GAME
        assert params.runs_allowed_per_game == lg.MLB_RUNS_PER_GAME
        assert params.team_era == lg.MLB_LEAGUE_ERA
        assert params.team_fip == lg.MLB_LEAGUE_FIP
        assert params.bullpen_era == lg.MLB_LEAGUE_ERA

    def test_missing_bullpen_era_falls_back_to_team_era(self) -> None:
        params = map_baseball_stats(self.make_stats(BaseballStats(team_era=3.2)))
        assert params.bullpen_era == 3.2


class TestRegistry:
    def test_mlb_spec(self) -> None:
        spec = get_plugin("mlb")
        assert spec.label == "baseball"
        assert spec.simulator is BaseballSimulator
        assert spec.map_team_stats is map_baseball_stats
        assert spec.grid_config == GridConfig(spread_radius=4, total_radius=6)
        assert spec.plugin_config == {"league_runs_per_game": 4.5}

    def test_ncaa_bsb_spec(self) -> None:
        spec = get_plugin("NCAA_BSB")
        assert spec.label == "baseball"
        assert spec.grid_config == BASEBALL_GRID_CONFIG
        assert spec.plugin_config == {"league_runs_per_game": 6.5}

    def test_sport_and_league_identity(self) -> None:
        sim = get_simulator("MLB")
        assert sim.get_sport() == "BASEBALL"
        assert sim.get_league() == "MLB"
        assert isinstance(sim, BaseballSimulator)
        sim.set_parameters(make_params(), make_params(), GameContext(league="NCAA_BSB"))
        assert sim.get_league() == "NCAA_BSB"  # adopted from the game context

    def test_league_config_override_wins(self) -> None:
        sim = BaseballSimulator({"league": "MLB"})
        sim.set_parameters(make_params(), make_params(), GameContext(league="NCAA_BSB"))
        assert sim.get_league() == "MLB"
