"""API-level football and hockey flows: NFL and NHL requests with stubbed stats."""

import uuid
from typing import Any

from httpx import Response


def envelope(data: dict[str, Any]) -> dict[str, Any]:
    return {"data": data, "meta": {"timestamp": "2026-07-06T12:00:00Z", "request_id": "req-test"}}


def game_payload(game_id: str, league: str, home_id: str, away_id: str) -> dict[str, Any]:
    return {
        "id": game_id,
        "league": league,
        "status": "SCHEDULED",
        "home_team": {"id": home_id, "name": "Home Team", "abbreviation": "HOM"},
        "away_team": {"id": away_id, "name": "Away Team", "abbreviation": "AWY"},
        "scheduled_start": "2026-09-13T17:00:00Z",
    }


def football_team_stats_payload(team_id: str, ppd_off: float, ppd_def: float) -> dict[str, Any]:
    return {
        "team_id": team_id,
        "team_abbreviation": "KC" if team_id == "team-kc" else "BUF",
        "season": 2026,
        "stats": {
            "football": {
                "points_per_game": ppd_off * 10.9,
                "points_allowed_per_game": ppd_def * 10.9,
                "drives_per_game": 10.9,
                "points_per_drive_off": ppd_off,
                "points_per_drive_def": ppd_def,
                "epa_per_play_off": 0.08,
                "epa_per_play_def": -0.02,
                "turnover_margin_per_game": 0.3,
            },
        },
    }


def hockey_team_stats_payload(team_id: str, gf: float, ga: float) -> dict[str, Any]:
    return {
        "team_id": team_id,
        "team_abbreviation": "COL" if team_id == "team-col" else "CHI",
        "season": 2026,
        "stats": {
            "hockey": {
                "goals_for_per_game": gf,
                "goals_against_per_game": ga,
                "shots_for_per_game": 31.0,
                "shots_against_per_game": 28.5,
                "power_play_pct": 0.235,
                "penalty_kill_pct": 0.805,
                "team_save_pct": 0.908,
            },
        },
    }


class TestFootballSimulationFlow:
    def mock_game(self, stats_service, game_id: str) -> None:
        stats_service.get(f"/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=envelope(game_payload(game_id, "NFL", "team-kc", "team-buf")))
        )
        stats_service.get("/api/v1/stats/teams/team-kc/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(football_team_stats_payload("team-kc", 2.35, 1.75)))
        )
        stats_service.get("/api/v1/stats/teams/team-buf/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(football_team_stats_payload("team-buf", 2.10, 1.90)))
        )

    def test_nfl_simulation_returns_football_distributions(self, client, stats_service) -> None:
        game_id = f"game-{uuid.uuid4()}"
        self.mock_game(stats_service, game_id)

        response = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert response.status_code == 201, response.text
        data = response.json()["data"]
        assert data["status"] == "completed"
        assert data["config"]["sport"] == "FOOTBALL"

        result = data["result"]
        # NFL overtime can leave rare ties, so a small draw probability is valid.
        assert 0.0 <= result["draw_probability"] < 0.03
        total_prob = result["home_win_probability"] + result["away_win_probability"] + result["draw_probability"]
        assert abs(total_prob - 1.0) < 0.001
        # Football-scale totals and the football line grids (radius 14/16).
        assert 30.0 < result["mean_total"] < 60.0
        assert len(result["spread_cover_probabilities"]) == 30
        assert len(result["total_over_probabilities"]) == 34

    def test_nfl_simulation_is_cached_by_parameters_hash(self, client, stats_service) -> None:
        game_id = f"game-{uuid.uuid4()}"
        self.mock_game(stats_service, game_id)
        body = {"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}}

        first = client.post("/api/v1/sim/simulations", json=body)
        assert first.status_code == 201, first.text
        assert first.json()["data"]["cached"] is False
        replay = client.post("/api/v1/sim/simulations", json=body)
        assert replay.json()["data"]["cached"] is True
        assert replay.json()["data"]["parameters_hash"] == first.json()["data"]["parameters_hash"]


class TestHockeySimulationFlow:
    def mock_game(self, stats_service, game_id: str) -> None:
        stats_service.get(f"/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=envelope(game_payload(game_id, "NHL", "team-col", "team-chi")))
        )
        stats_service.get("/api/v1/stats/teams/team-col/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(hockey_team_stats_payload("team-col", 3.5, 2.7)))
        )
        stats_service.get("/api/v1/stats/teams/team-chi/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(hockey_team_stats_payload("team-chi", 2.8, 3.3)))
        )

    def test_nhl_simulation_returns_two_way_probabilities(self, client, stats_service) -> None:
        game_id = f"game-{uuid.uuid4()}"
        self.mock_game(stats_service, game_id)

        response = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert response.status_code == 201, response.text
        data = response.json()["data"]
        assert data["status"] == "completed"
        assert data["config"]["sport"] == "HOCKEY"

        result = data["result"]
        # OT/SO resolution decides every game: no draw probability mass.
        assert result["draw_probability"] == 0.0
        total_prob = result["home_win_probability"] + result["away_win_probability"]
        assert abs(total_prob - 1.0) < 0.001
        # Hockey-scale totals and the hockey line grids (radius 3/4).
        assert 4.5 < result["mean_total"] < 8.5
        assert len(result["spread_cover_probabilities"]) == 8
        assert len(result["total_over_probabilities"]) == 10

        # Distributions carry no margin mass at zero (no ties in the final).
        run_id = data["simulation_run_id"]
        distributions = client.get(f"/api/v1/sim/simulations/{run_id}/distributions")
        assert distributions.status_code == 200
        margin = distributions.json()["data"]["distributions"]["margin"]
        assert margin["values"].get("0", 0.0) == 0.0
