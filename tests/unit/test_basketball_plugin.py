"""Statistical sanity tests for the vectorized basketball plugin.

Uses seeded RNGs and modest iteration counts to stay fast while keeping the
assertions statistically comfortable.
"""

import numpy as np
import pytest

from simulation_engine.core import league_averages as lg
from simulation_engine.core.params import GameContext, map_team_stats
from simulation_engine.core.plugins import PluginSpec, get_plugin, get_simulator
from simulation_engine.core.plugins.basketball import BasketballSimulator, _build_possession_pmf
from simulation_engine.core.runner import GridConfig

NEUTRAL = GameContext(neutral_site=True)
HOME = GameContext()
N = 5000


def simulate(home, away, context=NEUTRAL, seed=11, n=N, plugin_config=None):
    sim = BasketballSimulator(plugin_config or {})
    sim.set_parameters(home, away, context)
    rng = np.random.default_rng(seed)
    return sim.simulate_games(rng, n)


class TestPossessionModel:
    def test_pmf_sums_to_one(self, make_team_params) -> None:
        pmf = _build_possession_pmf(make_team_params("o"), make_team_params("d"))
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= 0).all()

    def test_calibration_anchors_expected_points_to_ratings(self, make_team_params) -> None:
        home = make_team_params("h", off_rating=118.0)
        away = make_team_params("a", def_rating=108.0)
        sim = BasketballSimulator({})
        sim.set_parameters(home, away, NEUTRAL)
        model, _ = sim._models()
        target = (118.0 + 108.0) / 2.0 / 100.0
        assert model.expected_points == pytest.approx(target, abs=0.002)


class TestGameOutcomes:
    def test_equal_teams_near_even_on_neutral_court(self, make_team_params) -> None:
        home, away = simulate(make_team_params("h"), make_team_params("a"), seed=5)
        win_prob = float(np.mean(home > away))
        assert win_prob == pytest.approx(0.5, abs=0.03)

    def test_higher_offensive_rating_wins_more(self, make_team_params) -> None:
        strong = make_team_params("h", off_rating=120.0)
        weak = make_team_params("a", off_rating=108.0)
        home, away = simulate(strong, weak)
        assert float(np.mean(home > away)) > 0.60
        assert float(np.mean(home)) > float(np.mean(away))

    def test_higher_pace_raises_totals(self, make_team_params) -> None:
        fast_h, fast_a = simulate(make_team_params("h", pace=104.0), make_team_params("a", pace=104.0))
        slow_h, slow_a = simulate(make_team_params("h", pace=94.0), make_team_params("a", pace=94.0))
        assert float(np.mean(fast_h + fast_a)) > float(np.mean(slow_h + slow_a)) + 5

    def test_home_court_advantage_shifts_margin(self, make_team_params) -> None:
        neutral_h, neutral_a = simulate(make_team_params("h"), make_team_params("a", off_rating=114.0), NEUTRAL)
        court_h, court_a = simulate(make_team_params("h"), make_team_params("a", off_rating=114.0), HOME)
        neutral_margin = float(np.mean(neutral_h - neutral_a))
        home_margin = float(np.mean(court_h - court_a))
        assert home_margin > neutral_margin + 0.5

    def test_home_advantage_configurable(self, make_team_params) -> None:
        big_h, big_a = simulate(
            make_team_params("h"), make_team_params("a"), HOME, plugin_config={"home_advantage": 6.0}
        )
        default_h, default_a = simulate(make_team_params("h"), make_team_params("a"), HOME)
        assert float(np.mean(big_h - big_a)) > float(np.mean(default_h - default_a)) + 1.0

    def test_no_draws_after_overtime(self, make_team_params) -> None:
        home, away = simulate(make_team_params("h"), make_team_params("a"))
        assert int(np.sum(home == away)) == 0

    def test_scores_in_plausible_nba_range(self, make_team_params) -> None:
        home, away = simulate(make_team_params("h"), make_team_params("a"))
        assert 100 < float(np.mean(home)) < 125
        assert 100 < float(np.mean(away)) < 125
        assert (home > 60).all() and (home < 180).all()


class TestPluginRegistry:
    def test_nba_supported(self) -> None:
        assert get_simulator("NBA").get_league() == "NBA"
        assert get_simulator("nba").get_sport() == "BASKETBALL"

    def test_unsupported_league_raises_422(self) -> None:
        from simulation_engine.api.errors import UnprocessableError

        with pytest.raises(UnprocessableError, match="not supported"):
            get_simulator("NCAA_HKY")

    def test_single_game_contract(self, make_team_params) -> None:
        sim = get_simulator("NBA")
        sim.set_parameters(make_team_params("h"), make_team_params("a"), NEUTRAL)
        result = sim.simulate_game(np.random.default_rng(1))
        assert result.home_score != result.away_score
        assert result.home_score > 60


class TestPluginSpecs:
    def test_nba_spec_fields(self) -> None:
        spec = get_plugin("nba")
        assert isinstance(spec, PluginSpec)
        assert spec.label == "basketball"
        assert spec.simulator is BasketballSimulator
        assert spec.map_team_stats is map_team_stats
        assert spec.grid_config == GridConfig(spread_radius=10, total_radius=12)
        assert spec.plugin_config == {}

    def test_unsupported_league_message_lists_supported(self) -> None:
        # NFL/NCAA_FB/NHL/NCAA_BB joined the supported list in Phase 6 Waves
        # 3-5; NCAA_HKY stays gated per ADR-026.
        from simulation_engine.api.errors import UnprocessableError

        supported = r"EPL, FIFA_WC, MLB, NBA, NCAA_BB, NCAA_BSB, NCAA_FB, NFL, NHL"
        with pytest.raises(
            UnprocessableError,
            match=rf"not supported for simulation \(supported: {supported}\)",
        ):
            get_plugin("NCAA_HKY")

    def test_shim_applies_request_plugin_config_over_defaults(self, make_team_params) -> None:
        sim = get_simulator("NBA", {"home_advantage": 6.0})
        assert isinstance(sim, BasketballSimulator)
        assert sim._home_advantage == 6.0


class TestNcaaBasketball:
    """Phase 6 Wave 5: NCAA_BB is a config-only entry over the NBA simulator."""

    def make_college_params(self, make_team_params, team_id: str, **overrides: float):
        college = {
            "pace": 68.0,
            "off_rating": 106.0,
            "def_rating": 103.0,
            "three_pct": lg.NCAA_BB_THREE_PCT,
            "ft_pct": lg.NCAA_BB_FT_PCT,
            "three_attempt_rate": lg.NCAA_BB_THREE_ATTEMPT_RATE,
        }
        college.update(overrides)
        return make_team_params(team_id, **college)

    def make_college_simulator(self, plugin_config: dict[str, object] | None = None) -> BasketballSimulator:
        spec = get_plugin("NCAA_BB")
        sim = spec.simulator({**spec.plugin_config, **(plugin_config or {})})
        assert isinstance(sim, BasketballSimulator)
        return sim

    def test_ncaa_bb_spec_fields(self) -> None:
        spec = get_plugin("ncaa_bb")
        assert spec.label == "basketball"
        assert spec.simulator is BasketballSimulator
        assert spec.map_team_stats is map_team_stats
        assert spec.grid_config == GridConfig(spread_radius=10, total_radius=12)
        assert spec.plugin_config == {
            "home_advantage": lg.NCAA_BB_HOME_ADVANTAGE,
            "league_avg_pace": lg.NCAA_BB_LEAGUE_AVG_PACE,
            "possession_std": lg.NCAA_BB_POSSESSION_STD,
            "possession_clip_min": lg.NCAA_BB_POSSESSION_CLIP_MIN,
            "possession_clip_max": lg.NCAA_BB_POSSESSION_CLIP_MAX,
            "ot_possession_fraction": lg.NCAA_BB_OT_POSSESSION_FRACTION,
        }

    def test_college_totals_land_near_145(self, make_team_params) -> None:
        sim = self.make_college_simulator()
        sim.set_parameters(
            self.make_college_params(make_team_params, "h"),
            self.make_college_params(make_team_params, "a"),
            GameContext(league="NCAA_BB", neutral_site=True),
        )
        home, away = sim.simulate_games(np.random.default_rng(17), N)
        total_mean = float(np.mean(home + away))
        # ~68 possessions at ~1.045 points per possession: total ~142.
        assert 130.0 < total_mean < 155.0
        assert int(np.sum(home == away)) == 0  # college overtime still decides games

    def test_college_possession_clip_bounds_apply(self, make_team_params) -> None:
        slow = self.make_college_simulator()
        slow.set_parameters(
            self.make_college_params(make_team_params, "h", pace=30.0),
            self.make_college_params(make_team_params, "a", pace=30.0),
            GameContext(league="NCAA_BB", neutral_site=True),
        )
        possessions = slow._draw_possessions(np.random.default_rng(3), 10_000)
        assert possessions.min() == lg.NCAA_BB_POSSESSION_CLIP_MIN  # pace 30 pins to the college floor

        fast = self.make_college_simulator()
        fast.set_parameters(
            self.make_college_params(make_team_params, "h", pace=200.0),
            self.make_college_params(make_team_params, "a", pace=200.0),
            GameContext(league="NCAA_BB", neutral_site=True),
        )
        possessions = fast._draw_possessions(np.random.default_rng(3), 10_000)
        assert possessions.max() == lg.NCAA_BB_POSSESSION_CLIP_MAX  # pace 200 pins to the college ceiling

    def test_nba_defaults_keep_pre_wave5_constants(self) -> None:
        sim = BasketballSimulator({})
        assert sim._home_advantage == lg.NBA_HOME_ADVANTAGE
        assert sim._league_avg_pace == lg.NBA_LEAGUE_AVG_PACE
        assert sim._possession_std == lg.NBA_POSSESSION_STD
        assert (sim._possession_clip_min, sim._possession_clip_max) == (70, 135)
        assert sim._ot_fraction == lg.NBA_OT_POSSESSION_FRACTION

    def test_stronger_college_home_advantage_widens_margin(self, make_team_params) -> None:
        home_p = self.make_college_params(make_team_params, "h")
        away_p = self.make_college_params(make_team_params, "a")
        college = self.make_college_simulator()
        college.set_parameters(home_p, away_p, GameContext(league="NCAA_BB"))
        nba_hca = self.make_college_simulator({"home_advantage": lg.NBA_HOME_ADVANTAGE})
        nba_hca.set_parameters(home_p, away_p, GameContext(league="NCAA_BB"))
        college_h, college_a = college.simulate_games(np.random.default_rng(23), N)
        nba_h, nba_a = nba_hca.simulate_games(np.random.default_rng(23), N)
        assert float(np.mean(college_h - college_a)) > float(np.mean(nba_h - nba_a)) + 0.3

    def test_league_identity_adopted_from_context(self, make_team_params) -> None:
        sim = get_simulator("NCAA_BB")
        assert sim.get_sport() == "BASKETBALL"
        assert sim.get_league() == "NBA"  # default before parameters arrive
        sim.set_parameters(
            self.make_college_params(make_team_params, "h"),
            self.make_college_params(make_team_params, "a"),
            GameContext(league="NCAA_BB"),
        )
        assert sim.get_league() == "NCAA_BB"
