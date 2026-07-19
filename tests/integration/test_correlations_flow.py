"""API-level correlations flow (Phase 7 Wave 1): soccer run, then GET /correlations.

Follows test_soccer_flow.py: FIFA_WC simulation against real Redis with a
stubbed statistics-service, then the correlation artifact is fetched without
and with an explicit ?legs= subset.
"""

import uuid
from typing import Any

import numpy as np
import redis
from httpx import Response


def envelope(data: dict[str, Any]) -> dict[str, Any]:
    return {"data": data, "meta": {"timestamp": "2026-07-19T12:00:00Z", "request_id": "req-test"}}


def soccer_game_payload(game_id: str, status: str = "SCHEDULED") -> dict[str, Any]:
    return {
        "id": game_id,
        "league": "FIFA_WC",
        "status": status,
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


def mock_soccer_game(stats_service, game_id: str) -> None:
    stats_service.get(f"/api/v1/stats/games/{game_id}").mock(
        return_value=Response(200, json=envelope(soccer_game_payload(game_id)))
    )
    stats_service.get("/api/v1/stats/teams/team-bra/stats", params={"stat_type": "all"}).mock(
        return_value=Response(200, json=envelope(soccer_team_stats_payload("team-bra", 1.25, 0.85)))
    )
    stats_service.get("/api/v1/stats/teams/team-ger/stats", params={"stat_type": "all"}).mock(
        return_value=Response(200, json=envelope(soccer_team_stats_payload("team-ger", 1.10, 0.95)))
    )


class TestCorrelationsFlow:
    def test_correlations_artifact_and_leg_subset(self, client, stats_service, redis_url) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_soccer_game(stats_service, game_id)

        created = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 42}},
        )
        assert created.status_code == 201, created.text
        run_id = created.json()["data"]["simulation_run_id"]

        # The artifact is persisted at run time under sim:correlations:{game_id}.
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        try:
            assert redis_client.exists(f"sim:correlations:{game_id}") == 1
        finally:
            redis_client.close()

        # Default artifact: full leg vocabulary, matrix aligned with legs, no joint.
        response = client.get(f"/api/v1/sim/simulations/{run_id}/correlations")
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["simulation_run_id"] == run_id
        assert data["game_id"] == game_id
        assert data["iterations"] == 2000
        legs = data["legs"]
        # Soccer has draws: all three moneyline legs plus the soccer line grids (8 spreads, 10 totals).
        assert {"MONEYLINE:HOME", "MONEYLINE:AWAY", "MONEYLINE:DRAW"} <= set(legs)
        assert len(legs) == 3 + 8 + 10
        assert set(data["marginals"]) == set(legs)
        matrix = data["matrix"]
        assert len(matrix) == len(legs)
        assert all(len(row) == len(legs) for row in matrix)
        assert all(matrix[i][i] == 1.0 for i in range(len(legs)))
        assert data["joint_probability"] is None
        # Soccer exposes the 13x13 analytic Dixon-Coles joint goal grid.
        grid = data["joint_goal_grid"]
        assert grid is not None
        assert len(grid) == 13
        assert all(len(row) == 13 for row in grid)
        assert abs(sum(sum(row) for row in grid) - 1.0) < 1e-6

        # Requested legs: subset marginals/matrix plus the empirical joint probability.
        subset = client.get(
            f"/api/v1/sim/simulations/{run_id}/correlations",
            params={"legs": "MONEYLINE:HOME,TOTAL:OVER:2.5"},
        )
        assert subset.status_code == 200, subset.text
        sub = subset.json()["data"]
        assert sub["legs"] == ["MONEYLINE:HOME", "TOTAL:OVER:2.5"]
        assert set(sub["marginals"]) == {"MONEYLINE:HOME", "TOTAL:OVER:2.5"}
        assert len(sub["matrix"]) == 2
        joint = sub["joint_probability"]
        assert joint is not None
        # The joint of an AND can never exceed either marginal.
        assert 0.0 <= joint <= min(sub["marginals"].values()) + 1e-9
        # Symmetric off-diagonal, consistent with the full artifact.
        assert sub["matrix"][0][1] == sub["matrix"][1][0]

        # Half-point complements resolve without being stored (AWAY/UNDER sides).
        complements = client.get(
            f"/api/v1/sim/simulations/{run_id}/correlations",
            params={"legs": "MONEYLINE:AWAY,TOTAL:UNDER:2.5,SPREAD:AWAY:0.5"},
        )
        assert complements.status_code == 200, complements.text
        comp = complements.json()["data"]
        assert comp["joint_probability"] is not None
        under_marginal = comp["marginals"]["TOTAL:UNDER:2.5"]
        over_marginal = data["marginals"]["TOTAL:OVER:2.5"]
        assert abs(under_marginal - (1.0 - over_marginal)) < 1e-6

        # Legs outside the stored vocabulary cannot be recomputed post-run: 422.
        unknown = client.get(
            f"/api/v1/sim/simulations/{run_id}/correlations",
            params={"legs": "SPREAD:HOME:-99.5"},
        )
        assert unknown.status_code == 422
        assert unknown.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"

    def test_correlations_joint_consistent_with_moneyline(self, client, stats_service) -> None:
        """Single-leg 'joint' equals that leg's marginal, and matches the run's home win probability."""
        game_id = f"game-{uuid.uuid4()}"
        mock_soccer_game(stats_service, game_id)
        created = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 2000, "random_seed": 7}},
        )
        assert created.status_code == 201, created.text
        run = created.json()["data"]
        run_id = run["simulation_run_id"]

        response = client.get(f"/api/v1/sim/simulations/{run_id}/correlations", params={"legs": "MONEYLINE:HOME"})
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert np.isclose(data["joint_probability"], data["marginals"]["MONEYLINE:HOME"], atol=1e-6)
        assert np.isclose(data["joint_probability"], run["result"]["home_win_probability"], atol=1e-3)
