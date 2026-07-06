"""API-level baseball simulation flow: MLB request with stubbed stats and probable pitchers."""

import uuid
from typing import Any

from httpx import Response


def envelope(data: dict[str, Any]) -> dict[str, Any]:
    return {"data": data, "meta": {"timestamp": "2026-07-05T12:00:00Z", "request_id": "req-test"}}


def baseball_game_payload(game_id: str, status: str = "SCHEDULED", with_pitchers: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": game_id,
        "league": "MLB",
        "status": status,
        "home_team": {"id": "team-nyy", "name": "New York Yankees", "abbreviation": "NYY"},
        "away_team": {"id": "team-bos", "name": "Boston Red Sox", "abbreviation": "BOS"},
        "scheduled_start": "2026-07-10T00:00:00Z",
    }
    if with_pitchers:
        payload["home_probable_pitcher"] = {
            "name": "Ace Homer",
            "external_id": "p-100",
            "throws": "R",
            "era": 2.90,
            "fip": 3.05,
            "k_bb_pct": 22.5,
            "innings_pitched": 110.2,
        }
        payload["away_probable_pitcher"] = {
            "name": "Wild Wally",
            "external_id": "p-200",
            "throws": "L",
            "era": 5.60,
            "fip": 5.85,
            "k_bb_pct": 6.0,
            "innings_pitched": 88.1,
        }
    return payload


def baseball_team_stats_payload(team_id: str, rs: float, ra: float) -> dict[str, Any]:
    return {
        "team_id": team_id,
        "team_abbreviation": "NYY" if team_id == "team-nyy" else "BOS",
        "season": 2026,
        "stats": {
            "baseball": {
                "runs_scored_per_game": rs,
                "runs_allowed_per_game": ra,
                "team_woba": 0.325,
                "team_obp": 0.330,
                "team_slg": 0.430,
                "batting_strikeout_pct": 21.0,
                "batting_walk_pct": 8.5,
                "team_era": 4.05,
                "team_fip": 4.00,
                "bullpen_era": 4.30,
            },
        },
    }


class TestBaseballSimulationFlow:
    def mock_game(self, stats_service, game_id: str, with_pitchers: bool = True) -> None:
        stats_service.get(f"/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=envelope(baseball_game_payload(game_id, with_pitchers=with_pitchers)))
        )
        stats_service.get("/api/v1/stats/teams/team-nyy/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(baseball_team_stats_payload("team-nyy", 5.1, 4.0)))
        )
        stats_service.get("/api/v1/stats/teams/team-bos/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(baseball_team_stats_payload("team-bos", 4.4, 4.7)))
        )

    def test_mlb_simulation_returns_two_way_probabilities(self, client, stats_service) -> None:
        game_id = f"game-{uuid.uuid4()}"
        self.mock_game(stats_service, game_id)

        response = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert response.status_code == 201, response.text
        data = response.json()["data"]
        assert data["status"] == "completed"
        assert data["config"]["sport"] == "BASEBALL"

        result = data["result"]
        # Extra innings decide every game: no draw probability mass.
        assert result["draw_probability"] == 0.0
        total_prob = result["home_win_probability"] + result["away_win_probability"]
        assert abs(total_prob - 1.0) < 0.001
        # Baseball-scale totals and the baseball line grids (radius 4/6).
        assert 5.0 < result["mean_total"] < 14.0
        assert len(result["spread_cover_probabilities"]) == 10
        assert len(result["total_over_probabilities"]) == 14
        # Default grids are half-point lines only, so no pushes are emitted.
        assert result["spread_push_probabilities"] == {}
        assert result["total_push_probabilities"] == {}

        # Distributions carry no margin mass at zero (no ties in baseball).
        run_id = data["simulation_run_id"]
        distributions = client.get(f"/api/v1/sim/simulations/{run_id}/distributions")
        assert distributions.status_code == 200
        margin = distributions.json()["data"]["distributions"]["margin"]
        assert margin["values"].get("0", 0.0) == 0.0

    def test_starter_announcement_invalidates_cached_simulation(self, client, stats_service) -> None:
        game_id = f"game-{uuid.uuid4()}"

        # First run: probable pitchers not yet announced.
        self.mock_game(stats_service, game_id, with_pitchers=False)
        first = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert first.status_code == 201, first.text
        assert first.json()["data"]["cached"] is False

        # Identical request replays from the parameters-hash cache.
        replay = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert replay.json()["data"]["cached"] is True
        assert replay.json()["data"]["parameters_hash"] == first.json()["data"]["parameters_hash"]

        # Starters get announced: the context hash changes, forcing a re-run.
        self.mock_game(stats_service, game_id, with_pitchers=True)
        announced = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert announced.status_code == 201, announced.text
        assert announced.json()["data"]["cached"] is False
        assert announced.json()["data"]["parameters_hash"] != first.json()["data"]["parameters_hash"]
