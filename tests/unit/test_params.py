"""Tests for statistics-service response -> plugin parameter mapping."""

import pytest

from simulation_engine.clients.statistics import (
    AdvancedStats,
    DefensiveStats,
    OffensiveStats,
    TeamStats,
)
from simulation_engine.core import league_averages as lg
from simulation_engine.core.params import map_team_stats


def make_stats(**overrides) -> TeamStats:
    offensive = OffensiveStats(
        points_per_game=114.5,
        field_goal_pct=0.48,
        three_point_pct=0.37,
        free_throw_pct=0.79,
        offensive_rating=116.2,
        pace=99.5,
        effective_fg_pct=0.55,
    )
    defensive = DefensiveStats(
        points_allowed_per_game=110.0,
        opponent_fg_pct=0.46,
        opponent_three_point_pct=0.35,
        defensive_rating=111.4,
    )
    advanced = AdvancedStats(net_rating=4.8, turnover_pct=12.5, offensive_rebound_pct=26.0)
    fields = {
        "team_id": "t-1",
        "team_abbreviation": "LAL",
        "offensive": offensive,
        "defensive": defensive,
        "advanced": advanced,
    }
    fields.update(overrides)
    return TeamStats(**fields)  # type: ignore[arg-type]


class TestMapTeamStats:
    def test_direct_mappings(self) -> None:
        params = map_team_stats(make_stats())
        assert params.team_id == "t-1"
        assert params.pace == 99.5
        assert params.off_rating == 116.2
        assert params.def_rating == 111.4
        assert params.three_pct == 0.37
        assert params.ft_pct == 0.79
        assert params.tov_pct == 12.5
        assert params.oreb_pct == 26.0

    def test_two_pct_derived_from_fg_mixture(self) -> None:
        params = map_team_stats(make_stats())
        # FG% = r*3P% + (1-r)*2P%  =>  2P% = (0.48 - 0.39*0.37) / 0.61
        expected = (0.48 - lg.NBA_THREE_ATTEMPT_RATE * 0.37) / (1 - lg.NBA_THREE_ATTEMPT_RATE)
        assert params.two_pct == pytest.approx(expected)

    def test_league_average_fallbacks_for_missing_contract_fields(self) -> None:
        params = map_team_stats(make_stats())
        assert params.three_attempt_rate == lg.NBA_THREE_ATTEMPT_RATE
        assert params.ft_rate == lg.NBA_FT_RATE
        assert params.forced_tov_pct == lg.NBA_TOV_PCT
        assert params.opp_oreb_pct == lg.NBA_OREB_PCT

    def test_zero_stats_fall_back_to_league_averages(self) -> None:
        empty = make_stats(
            offensive=OffensiveStats(),
            defensive=DefensiveStats(),
            advanced=AdvancedStats(),
        )
        params = map_team_stats(empty)
        assert params.pace == lg.NBA_LEAGUE_AVG_PACE
        assert params.three_pct == lg.NBA_THREE_PCT
        assert params.ft_pct == lg.NBA_FT_PCT
        assert params.tov_pct == lg.NBA_TOV_PCT
