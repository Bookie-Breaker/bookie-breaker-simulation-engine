"""API-level soccer simulation flow: FIFA_WC request with stubbed soccer stats."""

import uuid
from typing import Any

from httpx import Response


def envelope(data: dict[str, Any]) -> dict[str, Any]:
    return {"data": data, "meta": {"timestamp": "2026-07-05T12:00:00Z", "request_id": "req-test"}}


def soccer_game_payload(game_id: str, status: str = "SCHEDULED") -> dict[str, Any]:
    return {
        "id": game_id,
        "league": "FIFA_WC",
        "status": status,
        "home_team": {"id": "team-bra", "name": "Brazil", "abbreviation": "BRA"},
        "away_team": {"id": "team-ger", "name": "Germany", "abbreviation": "GER"},
        "scheduled_start": "2026-07-10T00:00:00Z",
    }


def soccer_team_stats_payload(team_id: str, attack: float, defense: float) -> dict[str, Any]:
    return {
        "team_id": team_id,
        "team_abbreviation": "BRA" if team_id == "team-bra" else "GER",
        "season": 2026,
        "stats": {
            "soccer": {
                "goals_for_per_match": round(1.35 * attack, 2),
                "goals_against_per_match": round(1.35 * defense, 2),
                "attack_strength": attack,
                "defense_strength": defense,
                "draws": 2,
                "form_goals_for_last5": 8.0,
                "form_goals_against_last5": 4.0,
                "form_points_last5": 11,
            },
        },
    }


class TestSoccerSimulationFlow:
    def test_fifa_wc_simulation_returns_three_way_probabilities(self, client, stats_service) -> None:
        game_id = f"game-{uuid.uuid4()}"
        stats_service.get(f"/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=envelope(soccer_game_payload(game_id)))
        )
        stats_service.get("/api/v1/stats/teams/team-bra/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(soccer_team_stats_payload("team-bra", 1.25, 0.85)))
        )
        stats_service.get("/api/v1/stats/teams/team-ger/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(soccer_team_stats_payload("team-ger", 1.10, 0.95)))
        )

        response = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert response.status_code == 201, response.text
        data = response.json()["data"]
        assert data["status"] == "completed"
        assert data["config"]["sport"] == "SOCCER"

        result = data["result"]
        # Draws are valid regulation outcomes (ADR-027): three-way split sums to 1.
        assert result["draw_probability"] > 0.0
        total_prob = result["home_win_probability"] + result["away_win_probability"] + result["draw_probability"]
        assert abs(total_prob - 1.0) < 0.001
        # Soccer-scale scores and the narrow soccer line grids (radius 3/4).
        assert 0.0 < result["mean_total"] < 6.0
        assert len(result["spread_cover_probabilities"]) == 8
        assert len(result["total_over_probabilities"]) == 10
        # Default grids are half-point lines only, so no pushes are emitted.
        assert result["spread_push_probabilities"] == {}
        assert result["total_push_probabilities"] == {}

        # Distributions include a margin mass at zero (the draw bucket).
        run_id = data["simulation_run_id"]
        distributions = client.get(f"/api/v1/sim/simulations/{run_id}/distributions")
        assert distributions.status_code == 200
        margin = distributions.json()["data"]["distributions"]["margin"]
        assert margin["values"]["0"] > 0.0
