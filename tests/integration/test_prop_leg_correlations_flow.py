"""API-level prop-leg correlations flow (Phase 7 Wave 4).

Follows test_player_props_flow.py: a props-enabled FIFA_WC simulation against
real Redis with a stubbed statistics-service, then GET /correlations is read
back with mixed team+player legs (the same-game-parlay payoff: an exact
empirical joint over the mix). A team-only run is asserted to store an
artifact without player legs, and player-leg requests against it 422.
"""

import uuid
from typing import Any

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


def player_detail(player_id: str, team_id: str, goals: int, shots: int, sot: int) -> dict[str, Any]:
    return {
        **player_summary(player_id, team_id),
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
]
AWAY_PLAYERS = [
    player_detail("ger-9", "team-ger", goals=11, shots=50, sot=24),
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
        return_value=Response(200, json=list_envelope([player_summary(p["id"], "team-bra") for p in HOME_PLAYERS]))
    )
    stats_service.get("/api/v1/stats/players", params={"league": "FIFA_WC", "team_id": "team-ger"}).mock(
        return_value=Response(200, json=list_envelope([player_summary(p["id"], "team-ger") for p in AWAY_PLAYERS]))
    )
    for detail in HOME_PLAYERS + AWAY_PLAYERS:
        stats_service.get(f"/api/v1/stats/players/{detail['id']}").mock(
            return_value=Response(200, json=envelope(detail))
        )


def create_run(client, game_id: str, include_player_props: bool) -> str:
    config: dict[str, Any] = {"iterations": 2000, "random_seed": 42}
    if include_player_props:
        config["include_player_props"] = True
    created = client.post("/api/v1/sim/simulations", json={"game_id": game_id, "config": config})
    assert created.status_code == 201, created.text
    run_id: str = created.json()["data"]["simulation_run_id"]
    return run_id


class TestPropLegCorrelationsFlow:
    def test_props_run_extends_vocabulary_and_serves_mixed_joint(self, client, stats_service) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_soccer_game(stats_service, game_id, with_players=True)
        run_id = create_run(client, game_id, include_player_props=True)

        # Default artifact: team legs plus PLAYER_PROP legs for both rosters.
        response = client.get(f"/api/v1/sim/simulations/{run_id}/correlations")
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        legs = data["legs"]
        player_legs = [leg for leg in legs if leg.startswith("PLAYER_PROP:")]
        assert player_legs, legs
        # Player legs are appended AFTER the team vocabulary (Wave 1 ordering preserved).
        assert legs[: len(legs) - len(player_legs)] == [leg for leg in legs if not leg.startswith("PLAYER_PROP:")]
        yes_leg = "PLAYER_PROP:bra-9:player_goal_scorer_anytime:YES"
        assert yes_leg in player_legs
        # Every roster player stores the YES leg and OVER legs on half-point lines only.
        assert "PLAYER_PROP:ger-9:player_goal_scorer_anytime:YES" in player_legs
        over_legs = [leg for leg in player_legs if ":OVER:" in leg]
        assert over_legs
        assert all(not float(leg.rsplit(":", 1)[1]).is_integer() for leg in over_legs)
        assert set(data["marginals"]) == set(legs)
        assert len(data["matrix"]) == len(legs)

        # Mixed team+player subset: the SGP payoff — an exact empirical joint.
        shots_over = next(leg for leg in over_legs if leg.startswith("PLAYER_PROP:bra-9:player_shots:OVER:"))
        mixed = ["MONEYLINE:HOME", yes_leg, shots_over]
        subset = client.get(
            f"/api/v1/sim/simulations/{run_id}/correlations",
            params={"legs": ",".join(mixed)},
        )
        assert subset.status_code == 200, subset.text
        sub = subset.json()["data"]
        assert sub["legs"] == mixed
        assert set(sub["marginals"]) == set(mixed)
        joint = sub["joint_probability"]
        assert joint is not None
        assert 0.0 <= joint <= min(sub["marginals"].values()) + 1e-9
        assert len(sub["matrix"]) == 3

        # NO resolves as the exact complement of the stored YES leg.
        no_leg = "PLAYER_PROP:bra-9:player_goal_scorer_anytime:NO"
        complements = client.get(
            f"/api/v1/sim/simulations/{run_id}/correlations",
            params={"legs": f"MONEYLINE:AWAY,{no_leg}"},
        )
        assert complements.status_code == 200, complements.text
        comp = complements.json()["data"]
        assert comp["joint_probability"] is not None
        assert abs(comp["marginals"][no_leg] - (1.0 - data["marginals"][yes_leg])) < 1e-6

        # UNDER on a stored half-point OVER line resolves as its complement.
        under_leg = shots_over.replace(":OVER:", ":UNDER:")
        under = client.get(f"/api/v1/sim/simulations/{run_id}/correlations", params={"legs": under_leg})
        assert under.status_code == 200, under.text
        under_marginal = under.json()["data"]["marginals"][under_leg]
        assert abs(under_marginal - (1.0 - data["marginals"][shots_over])) < 1e-6

        # Unknown players and malformed player legs 422 like team legs.
        unknown = client.get(
            f"/api/v1/sim/simulations/{run_id}/correlations",
            params={"legs": "PLAYER_PROP:nobody:player_shots:OVER:2.5"},
        )
        assert unknown.status_code == 422
        assert unknown.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"

    def test_team_only_run_stores_no_player_legs_and_422s_player_requests(self, client, stats_service) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_soccer_game(stats_service, game_id, with_players=False)
        run_id = create_run(client, game_id, include_player_props=False)

        response = client.get(f"/api/v1/sim/simulations/{run_id}/correlations")
        assert response.status_code == 200, response.text
        legs = response.json()["data"]["legs"]
        assert not any(leg.startswith("PLAYER_PROP:") for leg in legs)
        # Wave 1 team vocabulary is unchanged: 3 moneylines + soccer line grids.
        assert {"MONEYLINE:HOME", "MONEYLINE:AWAY", "MONEYLINE:DRAW"} <= set(legs)
        assert len(legs) == 3 + 8 + 10

        # Player legs on a run without player capture: 422 with a clear message.
        rejected = client.get(
            f"/api/v1/sim/simulations/{run_id}/correlations",
            params={"legs": "MONEYLINE:HOME,PLAYER_PROP:bra-9:player_goal_scorer_anytime:YES"},
        )
        assert rejected.status_code == 422
        error = rejected.json()["error"]
        assert error["code"] == "UNPROCESSABLE_ENTITY"
        assert "include_player_props" in error["message"]
