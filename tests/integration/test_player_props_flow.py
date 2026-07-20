"""API-level player-props flow (Phase 7 Wave 3): soccer run with include_player_props.

Follows test_correlations_flow.py: FIFA_WC simulation against real Redis with
a stubbed statistics-service (game + team stats + player endpoints), then the
player-distributions endpoint is read back with and without filters. A
props-off pregame run is asserted unaffected (no player blob, 404 read path).
"""

import uuid
from typing import Any

import redis
from httpx import Response


def envelope(data: Any) -> dict[str, Any]:
    return {"data": data, "meta": {"timestamp": "2026-07-19T12:00:00Z", "request_id": "req-test"}}


def list_envelope(data: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": data,
        "meta": {
            "timestamp": "2026-07-19T12:00:00Z",
            "request_id": "req-test",
            "pagination": {"limit": 200, "has_more": False, "next_cursor": ""},
        },
    }


def soccer_game_payload(game_id: str) -> dict[str, Any]:
    return {
        "id": game_id,
        "league": "FIFA_WC",
        "status": "SCHEDULED",
        "home_team": {"id": "team-bra", "name": "Brazil", "abbreviation": "BRA"},
        "away_team": {"id": "team-ger", "name": "Germany", "abbreviation": "GER"},
        "scheduled_start": "2026-07-24T00:00:00Z",
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


def player_summary(player_id: str, team_id: str, position: str = "F") -> dict[str, Any]:
    return {
        "id": player_id,
        "team_id": team_id,
        "first_name": "Player",
        "last_name": player_id.upper(),
        "position": position,
        "status": "ACTIVE",
    }


def player_detail(
    player_id: str, team_id: str, goals: int, shots: int, sot: int, position: str = "F"
) -> dict[str, Any]:
    return {
        **player_summary(player_id, team_id, position),
        "soccer_season_stats": {
            "appearances": 20,
            "minutes": 1700,
            "goals": goals,
            "assists": 3,
            "shots": shots,
            "shots_on_target": sot,
            "yellow_cards": 1,
            "red_cards": 0,
        },
    }


HOME_PLAYERS = [
    player_detail("bra-9", "team-bra", goals=14, shots=60, sot=30),
    player_detail("bra-10", "team-bra", goals=8, shots=45, sot=20),
    player_detail("bra-gk", "team-bra", goals=0, shots=0, sot=0, position="GK"),
]
AWAY_PLAYERS = [
    player_detail("ger-9", "team-ger", goals=11, shots=50, sot=24),
    player_detail("ger-8", "team-ger", goals=4, shots=30, sot=12),
]


def mock_soccer_game(stats_service, game_id: str, with_players: bool) -> None:
    stats_service.get(f"/api/v1/stats/games/{game_id}").mock(
        return_value=Response(200, json=envelope(soccer_game_payload(game_id)))
    )
    stats_service.get("/api/v1/stats/teams/team-bra/stats", params={"stat_type": "all"}).mock(
        return_value=Response(200, json=envelope(soccer_team_stats_payload("team-bra", 1.25, 0.85)))
    )
    stats_service.get("/api/v1/stats/teams/team-ger/stats", params={"stat_type": "all"}).mock(
        return_value=Response(200, json=envelope(soccer_team_stats_payload("team-ger", 1.10, 0.95)))
    )
    if not with_players:
        return
    stats_service.get("/api/v1/stats/players", params={"league": "FIFA_WC", "team_id": "team-bra"}).mock(
        return_value=Response(
            200, json=list_envelope([player_summary(p["id"], "team-bra", p["position"]) for p in HOME_PLAYERS])
        )
    )
    stats_service.get("/api/v1/stats/players", params={"league": "FIFA_WC", "team_id": "team-ger"}).mock(
        return_value=Response(
            200, json=list_envelope([player_summary(p["id"], "team-ger", p["position"]) for p in AWAY_PLAYERS])
        )
    )
    for detail in HOME_PLAYERS + AWAY_PLAYERS:
        stats_service.get(f"/api/v1/stats/players/{detail['id']}").mock(
            return_value=Response(200, json=envelope(detail))
        )


class TestPlayerPropsFlow:
    def test_props_run_stores_and_serves_player_distributions(self, client, stats_service, redis_url) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_soccer_game(stats_service, game_id, with_players=True)

        created = client.post(
            "/api/v1/sim/simulations",
            json={
                "game_id": game_id,
                "config": {"iterations": 2000, "random_seed": 42, "include_player_props": True},
            },
        )
        assert created.status_code == 201, created.text
        run = created.json()["data"]
        run_id = run["simulation_run_id"]

        # The player blob is persisted at run time under sim:player_distributions:{game_id}.
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        try:
            assert redis_client.exists(f"sim:player_distributions:{game_id}") == 1
        finally:
            redis_client.close()

        response = client.get(f"/api/v1/sim/simulations/{run_id}/player-distributions")
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["simulation_run_id"] == run_id
        assert data["game_id"] == game_id
        players = data["players"]
        # Field players from both rosters are present; the goalkeeper is excluded.
        assert set(players) == {"bra-9", "bra-10", "ger-9", "ger-8"}
        assert players["bra-9"]["team"] == "HOME"
        assert players["ger-9"]["team"] == "AWAY"

        striker = players["bra-9"]["stats"]
        assert set(striker) == {"player_goal_scorer_anytime", "player_shots", "player_shots_on_target"}
        anytime = striker["player_goal_scorer_anytime"]
        assert anytime["yes_probability"] is not None
        assert 0.0 < anytime["yes_probability"] < 1.0
        assert anytime["over_probabilities"] is None
        shots = striker["player_shots"]
        assert shots["yes_probability"] is None
        lines = sorted(float(line) for line in shots["over_probabilities"])
        probs = [shots["over_probabilities"][f"{line:.1f}"] for line in lines]
        assert all(a >= b for a, b in zip(probs, probs[1:], strict=False))
        distribution = shots["distribution"]
        assert abs(sum(distribution["values"].values()) - 1.0) < 0.01

        # The better scorer carries the higher anytime probability.
        assert (
            players["bra-9"]["stats"]["player_goal_scorer_anytime"]["yes_probability"]
            > players["bra-10"]["stats"]["player_goal_scorer_anytime"]["yes_probability"]
        )

        # Filters narrow the payload; unknown filters 404.
        filtered = client.get(
            f"/api/v1/sim/simulations/{run_id}/player-distributions",
            params={"player_id": "ger-9", "stat_type": "player_shots"},
        )
        assert filtered.status_code == 200, filtered.text
        narrowed = filtered.json()["data"]["players"]
        assert set(narrowed) == {"ger-9"}
        assert set(narrowed["ger-9"]["stats"]) == {"player_shots"}
        assert (
            client.get(
                f"/api/v1/sim/simulations/{run_id}/player-distributions", params={"player_id": "nobody"}
            ).status_code
            == 404
        )

    def test_pregame_run_unaffected_and_read_path_404s_without_props(self, client, stats_service, redis_url) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_soccer_game(stats_service, game_id, with_players=False)

        created = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert created.status_code == 201, created.text
        run = created.json()["data"]
        run_id = run["simulation_run_id"]
        assert run["result"]["home_win_probability"] > 0.0

        # Props off: no player blob is written and the read path 404s.
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        try:
            assert redis_client.exists(f"sim:player_distributions:{game_id}") == 0
        finally:
            redis_client.close()
        response = client.get(f"/api/v1/sim/simulations/{run_id}/player-distributions")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
