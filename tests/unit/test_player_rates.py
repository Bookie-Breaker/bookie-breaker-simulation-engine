"""Tests for statistics-service player detail -> PlayerRates mapping (Phase 7 Wave 3)."""

import pytest

from simulation_engine.clients.statistics import PlayerDetail, PlayerSeasonStats, SoccerSeasonStats
from simulation_engine.core import league_averages as lg
from simulation_engine.core.player_rates import YES_NO_STAT_KEYS, build_player_rates


def soccer_player(
    player_id: str,
    goals: int = 5,
    shots: int = 40,
    shots_on_target: int = 18,
    appearances: int = 20,
    minutes: int = 1800,
    position: str = "F",
    status: str = "ACTIVE",
) -> PlayerDetail:
    return PlayerDetail(
        id=player_id,
        first_name="Player",
        last_name=player_id.upper(),
        position=position,
        status=status,
        soccer_season_stats=SoccerSeasonStats(
            appearances=appearances,
            minutes=minutes,
            goals=goals,
            assists=3,
            shots=shots,
            shots_on_target=shots_on_target,
        ),
    )


def nba_player(
    player_id: str,
    points: float = 22.0,
    rebounds: float = 6.0,
    assists: float = 5.0,
    minutes: float = 34.0,
    three_pct: float = 0.37,
    games: int = 60,
    status: str = "ACTIVE",
) -> PlayerDetail:
    return PlayerDetail(
        id=player_id,
        first_name="Player",
        last_name=player_id.upper(),
        position="G",
        status=status,
        season_stats=PlayerSeasonStats(
            season=2026,
            games_played=games,
            minutes_per_game=minutes,
            points_per_game=points,
            rebounds_per_game=rebounds,
            assists_per_game=assists,
            three_point_pct=three_pct,
        ),
    )


class TestSoccerRates:
    def test_goal_shares_normalize_to_one(self) -> None:
        rates = build_player_rates("soccer", [soccer_player("p1", goals=10), soccer_player("p2", goals=2)], "HOME")
        assert len(rates) == 2
        assert sum(p.rates["goal_share"] for p in rates) == pytest.approx(1.0)
        by_id = {p.player_id: p for p in rates}
        assert by_id["p1"].rates["goal_share"] > by_id["p2"].rates["goal_share"]
        assert all(p.team == "HOME" for p in rates)

    def test_smoothing_keeps_zero_goal_players_nonzero(self) -> None:
        rates = build_player_rates("soccer", [soccer_player("star", goals=10), soccer_player("sub", goals=0)], "AWAY")
        by_id = {p.player_id: p for p in rates}
        assert by_id["sub"].rates["goal_share"] > 0.0

    def test_minutes_weighting_downweights_part_timers(self) -> None:
        full = soccer_player("full", goals=3, minutes=1800, appearances=20)
        part = soccer_player("part", goals=3, minutes=450, appearances=20)
        rates = {p.player_id: p for p in build_player_rates("soccer", [full, part], "HOME")}
        assert rates["full"].rates["goal_share"] > rates["part"].rates["goal_share"]
        assert rates["part"].rates["minutes_share"] == pytest.approx(450 / (20 * 90))

    def test_shot_rates_are_per_appearance(self) -> None:
        rates = build_player_rates("soccer", [soccer_player("p1", shots=40, shots_on_target=18)], "HOME")
        assert rates[0].rates["shots_per_match"] == pytest.approx(2.0)
        assert rates[0].rates["sot_per_match"] == pytest.approx(0.9)

    def test_goalkeepers_and_out_players_excluded(self) -> None:
        players = [
            soccer_player("gk", position="GK"),
            soccer_player("out", status="OUT"),
            soccer_player("f1"),
        ]
        rates = build_player_rates("soccer", players, "HOME")
        assert [p.player_id for p in rates] == ["f1"]

    def test_players_without_soccer_stats_skipped(self) -> None:
        bare = PlayerDetail(id="bare", position="F", status="ACTIVE")
        zero_apps = soccer_player("zero", appearances=0)
        assert build_player_rates("soccer", [bare, zero_apps], "HOME") == []


class TestBasketballRates:
    def test_points_weight_is_ppg_times_minutes_share(self) -> None:
        rates = build_player_rates("basketball", [nba_player("p1", points=24.0, minutes=36.0)], "HOME")
        assert rates[0].rates["points_weight"] == pytest.approx(24.0 * 36.0 / 48.0)
        assert rates[0].rates["rebounds_per_game"] == pytest.approx(6.0)
        assert rates[0].rates["assists_per_game"] == pytest.approx(5.0)

    def test_threes_estimate_zero_for_non_shooters(self) -> None:
        rates = build_player_rates("basketball", [nba_player("big", three_pct=0.0)], "HOME")
        assert rates[0].rates["threes_per_game"] == 0.0

    def test_threes_estimate_scales_with_shooting_pct(self) -> None:
        sniper = build_player_rates("basketball", [nba_player("s", three_pct=0.42)], "HOME")[0]
        league = build_player_rates("basketball", [nba_player("l", three_pct=lg.NBA_THREE_PCT)], "HOME")[0]
        assert sniper.rates["threes_per_game"] > league.rates["threes_per_game"] > 0.0

    def test_out_and_statless_players_excluded(self) -> None:
        players = [
            nba_player("out", status="OUT"),
            PlayerDetail(id="bare", position="C", status="ACTIVE"),
            nba_player("zero-games", games=0),
            nba_player("p1"),
        ]
        rates = build_player_rates("basketball", players, "AWAY")
        assert [p.player_id for p in rates] == ["p1"]

    def test_all_zero_weights_yields_empty(self) -> None:
        rates = build_player_rates("basketball", [nba_player("p1", points=0.0)], "HOME")
        assert rates == []


class TestDormantSports:
    def test_unknown_plugin_labels_return_empty(self) -> None:
        players = [nba_player("p1")]
        for label in ("baseball", "football", "hockey"):
            assert build_player_rates(label, players, "HOME") == []

    def test_empty_roster_returns_empty(self) -> None:
        assert build_player_rates("soccer", [], "HOME") == []
        assert build_player_rates("basketball", [], "AWAY") == []


class TestCanonicalKeys:
    def test_yes_no_keys_are_the_adr029_yes_no_markets(self) -> None:
        assert {"player_goal_scorer_anytime", "player_anytime_td"} == YES_NO_STAT_KEYS
