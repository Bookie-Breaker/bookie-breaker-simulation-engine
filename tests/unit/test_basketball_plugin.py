"""Statistical sanity tests for the vectorized basketball plugin.

Uses seeded RNGs and modest iteration counts to stay fast while keeping the
assertions statistically comfortable.
"""

import numpy as np
import pytest

from simulation_engine.core.params import GameContext
from simulation_engine.core.plugins import get_simulator
from simulation_engine.core.plugins.basketball import BasketballSimulator, _build_possession_pmf

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
            get_simulator("NFL")

    def test_single_game_contract(self, make_team_params) -> None:
        sim = get_simulator("NBA")
        sim.set_parameters(make_team_params("h"), make_team_params("a"), NEUTRAL)
        result = sim.simulate_game(np.random.default_rng(1))
        assert result.home_score != result.away_score
        assert result.home_score > 60
