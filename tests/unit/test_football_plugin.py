"""Football plugin tests: calibration exactness, key numbers, overtime formats.

The drive-outcome PMFs are exact discrete distributions, so calibration
assertions are analytic (1e-6 mean matching, P(score) band, TD:FG ratio);
game-level assertions (key-number mass, tie rates, HFA shift) use seeded
large samples so they are deterministic given the seed.
"""

import time

import numpy as np
import pytest

from simulation_engine.clients.statistics import (
    AdvancedStats,
    DefensiveStats,
    FootballStats,
    OffensiveStats,
    TeamStats,
)
from simulation_engine.core import league_averages as lg
from simulation_engine.core.params import GameContext
from simulation_engine.core.plugins import FOOTBALL_GRID_CONFIG, get_plugin, get_simulator
from simulation_engine.core.plugins.football import (
    FootballParams,
    FootballSimulator,
    _drive_outcome_pmf,
    _drive_pmf,
    _td_fg_ratio,
    map_football_stats,
)
from simulation_engine.core.runner import GridConfig, run_monte_carlo

NFL_NEUTRAL = GameContext(league="NFL", neutral_site=True)
NFL_HOME = GameContext(league="NFL")
POINTS = np.arange(8)
N = 200_000
TARGET_RANGE = np.linspace(1.0, 3.2, 45)


def make_params(
    ppd_off: float = lg.NFL_POINTS_PER_DRIVE,
    ppd_def: float = lg.NFL_POINTS_PER_DRIVE,
    drives: float = lg.NFL_DRIVES_PER_TEAM_MU,
) -> FootballParams:
    return FootballParams(
        points_per_game=ppd_off * drives,
        points_allowed_per_game=ppd_def * drives,
        drives_per_game=drives,
        points_per_drive_off=ppd_off,
        points_per_drive_def=ppd_def,
        epa_per_play_off=0.0,
        epa_per_play_def=0.0,
    )


def make_simulator(
    home: FootballParams | None = None,
    away: FootballParams | None = None,
    context: GameContext = NFL_NEUTRAL,
    league: str = "NFL",
) -> FootballSimulator:
    spec = get_plugin(league)
    sim = spec.simulator(dict(spec.plugin_config))
    assert isinstance(sim, FootballSimulator)
    sim.set_parameters(home or make_params(), away or make_params(), context)
    return sim


def pmf_mean(pmf: np.ndarray) -> float:
    return float(np.dot(pmf, POINTS))


class TestDriveOutcomeDistribution:
    def test_pmf_shape_and_support(self) -> None:
        pmf = _drive_outcome_pmf(0.4, 1.35)
        assert pmf.shape == (8,)
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= 0).all()
        # Mass only on the {0, 3, 7} quantization; safeties/2pt are approximations.
        assert pmf[[1, 2, 4, 5, 6]] == pytest.approx([0.0] * 5)
        assert pmf[0] == pytest.approx(0.6)
        assert pmf[7] / pmf[3] == pytest.approx(1.35)

    @pytest.mark.parametrize("league_ppd", [lg.NFL_POINTS_PER_DRIVE, lg.NCAA_FB_POINTS_PER_DRIVE])
    def test_calibrated_mean_matches_target_to_1e6_across_range(self, league_ppd: float) -> None:
        for target in TARGET_RANGE:
            pmf = _drive_pmf(float(target), league_ppd)
            assert abs(pmf_mean(pmf) - float(target)) < 1e-6, f"target {target}"

    @pytest.mark.parametrize("league_ppd", [lg.NFL_POINTS_PER_DRIVE, lg.NCAA_FB_POINTS_PER_DRIVE])
    def test_score_probability_stays_in_realistic_band(self, league_ppd: float) -> None:
        for target in TARGET_RANGE:
            pmf = _drive_pmf(float(target), league_ppd)
            score_prob = 1.0 - pmf[0]
            assert 0.25 <= score_prob <= 0.55, f"target {target}: P(score)={score_prob}"

    @pytest.mark.parametrize("league_ppd", [lg.NFL_POINTS_PER_DRIVE, lg.NCAA_FB_POINTS_PER_DRIVE])
    def test_td_fg_ratio_at_league_average_and_monotone(self, league_ppd: float) -> None:
        assert _td_fg_ratio(league_ppd, league_ppd) == pytest.approx(lg.FOOTBALL_TD_FG_RATIO_BASE)
        pmf = _drive_pmf(league_ppd, league_ppd)
        assert pmf[7] / pmf[3] == pytest.approx(lg.FOOTBALL_TD_FG_RATIO_BASE)
        # Hotter offenses convert more of their scores into touchdowns.
        ratios = [_td_fg_ratio(float(t), league_ppd) for t in TARGET_RANGE]
        assert all(a <= b for a, b in zip(ratios[:-1], ratios[1:], strict=True))

    def test_target_clamped_to_calibratable_range(self) -> None:
        low = _drive_pmf(0.2, lg.NFL_POINTS_PER_DRIVE)
        high = _drive_pmf(5.0, lg.NFL_POINTS_PER_DRIVE)
        assert pmf_mean(low) == pytest.approx(1.0, abs=1e-6)
        assert pmf_mean(high) == pytest.approx(3.2, abs=1e-6)


class TestParametersAndTargets:
    def test_even_matchup_totals_hit_league_rate(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(11), N)
        expected_total = 2.0 * lg.NFL_POINTS_PER_DRIVE * lg.NFL_DRIVES_PER_TEAM_MU
        # Overtime adds a small premium on top of the regulation expectation.
        assert float((home + away).mean()) == pytest.approx(expected_total, abs=1.0)

    def test_odds_ratio_blend_shifts_means(self) -> None:
        strong = make_simulator(home=make_params(ppd_off=2.4), away=make_params(ppd_def=2.2))
        even = make_simulator()
        strong_h, _ = strong.simulate_games(np.random.default_rng(13), N)
        even_h, _ = even.simulate_games(np.random.default_rng(13), N)
        # 2.4 x 2.2 / 1.95 = 2.708 points per drive vs the 1.95 baseline.
        assert float(strong_h.mean()) > float(even_h.mean()) + 6.0

    def test_home_field_advantage_shifts_margin_by_hfa_points(self) -> None:
        neutral = make_simulator(context=NFL_NEUTRAL)
        home_field = make_simulator(context=NFL_HOME)
        neutral_h, neutral_a = neutral.simulate_games(np.random.default_rng(17), N)
        court_h, court_a = home_field.simulate_games(np.random.default_rng(17), N)
        neutral_margin = float((neutral_h - neutral_a).mean())
        home_margin = float((court_h - court_a).mean())
        assert neutral_margin == pytest.approx(0.0, abs=0.15)
        assert home_margin - neutral_margin == pytest.approx(lg.NFL_HFA_MARGIN_POINTS, abs=0.25)

    def test_drive_counts_clipped_to_league_bounds(self) -> None:
        sim = make_simulator()
        counts = sim._draw_drive_counts(np.random.default_rng(3), 50_000)
        assert counts.min() >= lg.NFL_DRIVES_CLIP_MIN
        assert counts.max() <= lg.NFL_DRIVES_CLIP_MAX
        assert float(counts.mean()) == pytest.approx(lg.NFL_DRIVES_PER_TEAM_MU, abs=0.05)

    def test_overtime_pmf_is_elevated(self) -> None:
        sim = make_simulator()
        home, _ = sim._models()
        assert pmf_mean(home.overtime_pmf) > pmf_mean(home.regulation_pmf) * 1.2

    def test_requires_football_params(self, make_team_params) -> None:
        sim = FootballSimulator({})
        with pytest.raises(TypeError, match="FootballParams"):
            sim.set_parameters(make_team_params("h"), make_team_params("a"), NFL_NEUTRAL)

    def test_simulating_before_set_parameters_raises(self) -> None:
        with pytest.raises(RuntimeError, match="set_parameters"):
            FootballSimulator({}).simulate_games(np.random.default_rng(1), 10)


class TestKeyNumbers:
    def test_margin_mass_concentrates_on_3_and_7(self) -> None:
        """Key numbers must emerge from the score quantization: local mass at
        |margin| in {3, 7} exceeds each neighbor in {2, 4} / {6, 8}."""
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(7), N)
        abs_margin = np.abs(home - away)
        mass = {k: float(np.mean(abs_margin == k)) for k in (2, 3, 4, 6, 7, 8)}
        assert mass[3] / mass[2] > 1.0
        assert mass[3] / mass[4] > 1.0
        assert mass[7] / mass[6] > 1.0
        assert mass[7] / mass[8] > 1.0


class TestOvertime:
    def test_nfl_tie_rate_in_band_at_even_matchup(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(19), N)
        tie_rate = float(np.mean(home == away))
        assert 0.003 <= tie_rate <= 0.01  # NFL regular-season ties are rare but real
        assert sim._last_standing_ties == int(np.sum(home == away))
        # Most regulation ties get resolved by the OT exchange + sudden death
        # (each round leaves ~40% tied, so ~1/6 of regulation ties survive).
        assert sim._last_regulation_ties > 5 * sim._last_standing_ties

    def test_ncaa_never_ties(self) -> None:
        params = make_params(ppd_off=lg.NCAA_FB_POINTS_PER_DRIVE, ppd_def=lg.NCAA_FB_POINTS_PER_DRIVE, drives=12.5)
        sim = make_simulator(params, params, GameContext(league="NCAA_FB", neutral_site=True), league="NCAA_FB")
        home, away = sim.simulate_games(np.random.default_rng(23), N)
        assert int(np.sum(home == away)) == 0
        assert sim._last_standing_ties == 0
        assert sim._last_forced_tiebreaks == 0  # alternating rounds decide well before the cap
        assert sim._last_regulation_ties > 0  # ties happened, and were all resolved

    def test_even_matchup_is_near_coin_flip(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(29), N)
        decided = home != away
        assert float(np.mean(home[decided] > away[decided])) == pytest.approx(0.5, abs=0.01)


class TestSampling:
    def test_nfl_scores_in_plausible_range(self) -> None:
        sim = make_simulator()
        home, away = sim.simulate_games(np.random.default_rng(31), 50_000)
        assert 40.0 < float((home + away).mean()) < 46.0
        assert home.max() <= 16 * 7 + 14  # regulation cap plus overtime
        assert home.min() >= 0
        assert home.dtype == np.int32 and away.dtype == np.int32

    def test_ncaa_fb_scores_higher_than_nfl(self) -> None:
        college = make_params(ppd_off=2.15, ppd_def=2.15, drives=12.5)
        ncaa = make_simulator(college, college, GameContext(league="NCAA_FB", neutral_site=True), league="NCAA_FB")
        nfl = make_simulator()
        ncaa_h, ncaa_a = ncaa.simulate_games(np.random.default_rng(37), 50_000)
        nfl_h, nfl_a = nfl.simulate_games(np.random.default_rng(37), 50_000)
        assert float((ncaa_h + ncaa_a).mean()) > float((nfl_h + nfl_a).mean()) + 8.0

    def test_single_game_contract(self) -> None:
        result = make_simulator().simulate_game(np.random.default_rng(5))
        assert result.home_score >= 0
        assert result.away_score >= 0

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
    def run(self, seed: int = 11, league: str = "NFL", context: GameContext = NFL_HOME):
        sim = get_simulator(league)
        return run_monte_carlo(
            sim,
            make_params(ppd_off=2.1),
            make_params(ppd_def=1.8),
            context,
            iterations=50_000,
            convergence_threshold=1e-9,
            seed=seed,
            grid_config=FOOTBALL_GRID_CONFIG,
        )

    def test_nfl_ties_surface_as_draw_probability(self) -> None:
        out = self.run()
        assert 0.0 < out.draw_prob < 0.02  # rare NFL ties grade two-way moneylines as PUSH
        assert out.home_win_prob + out.away_win_prob + out.draw_prob == pytest.approx(1.0)

    def test_ncaa_fb_has_no_draw_probability(self) -> None:
        out = self.run(league="NCAA_FB", context=GameContext(league="NCAA_FB"))
        assert out.draw_prob == 0.0
        assert out.home_win_prob + out.away_win_prob == pytest.approx(1.0)

    def test_football_grid_sizes(self) -> None:
        out = self.run()
        assert len(out.spread_covers) == 2 * FOOTBALL_GRID_CONFIG.spread_radius + 2
        assert len(out.total_overs) == 2 * FOOTBALL_GRID_CONFIG.total_radius + 2
        assert all(not float(line).is_integer() for line in out.spread_covers)

    def test_same_seed_identical_runs(self) -> None:
        a = self.run(seed=99)
        b = self.run(seed=99)
        assert np.array_equal(a.home_scores, b.home_scores)
        assert np.array_equal(a.away_scores, b.away_scores)


class TestMapFootballStats:
    def make_stats(self, football: FootballStats) -> TeamStats:
        return TeamStats(
            team_id="t-fb",
            team_abbreviation="KC",
            offensive=OffensiveStats(),
            defensive=DefensiveStats(),
            advanced=AdvancedStats(),
            football=football,
        )

    def test_populated_block_maps_directly(self) -> None:
        params = map_football_stats(
            self.make_stats(
                FootballStats(
                    points_per_game=27.4,
                    points_allowed_per_game=19.1,
                    drives_per_game=11.3,
                    points_per_drive_off=2.42,
                    points_per_drive_def=1.69,
                    epa_per_play_off=0.12,
                    epa_per_play_def=-0.05,
                    turnover_margin_per_game=0.4,
                    sp_plus_rating=0.0,
                )
            )
        )
        assert params == FootballParams(
            points_per_game=27.4,
            points_allowed_per_game=19.1,
            drives_per_game=11.3,
            points_per_drive_off=2.42,
            points_per_drive_def=1.69,
            epa_per_play_off=0.12,
            epa_per_play_def=-0.05,
        )

    def test_empty_block_falls_back_to_nfl_league_averages(self) -> None:
        params = map_football_stats(self.make_stats(FootballStats()))
        assert params.drives_per_game == lg.NFL_DRIVES_PER_TEAM_MU
        assert params.points_per_drive_off == lg.NFL_POINTS_PER_DRIVE
        assert params.points_per_drive_def == lg.NFL_POINTS_PER_DRIVE
        assert params.points_per_game == pytest.approx(lg.NFL_POINTS_PER_DRIVE * lg.NFL_DRIVES_PER_TEAM_MU)
        assert params.epa_per_play_off == 0.0  # league-average EPA


class TestRegistry:
    def test_nfl_spec(self) -> None:
        spec = get_plugin("nfl")
        assert spec.label == "football"
        assert spec.simulator is FootballSimulator
        assert spec.map_team_stats is map_football_stats
        assert spec.grid_config == GridConfig(spread_radius=14, total_radius=16)
        assert spec.plugin_config == {
            "drives_mu": 10.9,
            "drives_sigma": 1.2,
            "drives_clip_min": 7,
            "drives_clip_max": 16,
            "league_points_per_drive": 1.95,
            "hfa_margin_points": 2.2,
            "ot_ties_allowed": True,
        }

    def test_ncaa_fb_spec(self) -> None:
        spec = get_plugin("NCAA_FB")
        assert spec.label == "football"
        assert spec.grid_config == FOOTBALL_GRID_CONFIG
        assert spec.plugin_config == {
            "drives_mu": 12.5,
            "drives_sigma": 1.6,
            "drives_clip_min": 8,
            "drives_clip_max": 18,
            "league_points_per_drive": 2.15,
            "hfa_margin_points": 3.0,
            "ot_ties_allowed": False,
        }

    def test_sport_and_league_identity(self) -> None:
        sim = get_simulator("NFL")
        assert sim.get_sport() == "FOOTBALL"
        assert sim.get_league() == "NFL"
        assert isinstance(sim, FootballSimulator)
        sim.set_parameters(make_params(), make_params(), GameContext(league="NCAA_FB"))
        assert sim.get_league() == "NCAA_FB"  # adopted from the game context

    def test_league_config_override_wins(self) -> None:
        sim = FootballSimulator({"league": "NFL"})
        sim.set_parameters(make_params(), make_params(), GameContext(league="NCAA_FB"))
        assert sim.get_league() == "NFL"
