"""Integration fixtures: one session-scoped Redis container, mocked statistics-service.

Kept deliberately light (single container, 2000-iteration simulations) so the
pre-push hook stays fast on modest hardware.
"""

from collections.abc import Iterator
from typing import Any

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response
from testcontainers.redis import RedisContainer

from simulation_engine.config import Settings
from simulation_engine.main import create_app

STATS_URL = "http://stats.test"


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    with RedisContainer("redis:7-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"


@pytest.fixture
def client(redis_url: str) -> Iterator[TestClient]:
    settings = Settings(
        redis_url=redis_url,
        statistics_service_url=STATS_URL,
        simulation_iterations=2000,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def game_payload(game_id: str, status: str = "SCHEDULED") -> dict[str, Any]:
    return {
        "id": game_id,
        "league": "NBA",
        "status": status,
        "home_team": {"id": "team-home", "name": "Los Angeles Lakers", "abbreviation": "LAL"},
        "away_team": {"id": "team-away", "name": "Boston Celtics", "abbreviation": "BOS"},
        "scheduled_start": "2026-07-05T00:00:00Z",
    }


def team_stats_payload(team_id: str, off_rating: float, def_rating: float) -> dict[str, Any]:
    return {
        "team_id": team_id,
        "team_abbreviation": "LAL" if team_id == "team-home" else "BOS",
        "season": 2026,
        "stats": {
            "offensive": {
                "points_per_game": 114.0,
                "field_goal_pct": 0.48,
                "three_point_pct": 0.37,
                "free_throw_pct": 0.79,
                "offensive_rating": off_rating,
                "pace": 99.5,
                "effective_fg_pct": 0.55,
            },
            "defensive": {
                "points_allowed_per_game": 110.0,
                "opponent_fg_pct": 0.46,
                "opponent_three_point_pct": 0.35,
                "defensive_rating": def_rating,
            },
            "advanced": {"net_rating": 4.0, "turnover_pct": 12.5, "offensive_rebound_pct": 26.0},
        },
    }


def envelope(data: dict[str, Any]) -> dict[str, Any]:
    return {"data": data, "meta": {"timestamp": "2026-07-04T12:00:00Z", "request_id": "req-test"}}


@pytest.fixture
def stats_service() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=STATS_URL, assert_all_called=False) as router:
        yield router


@pytest.fixture
def mock_game(stats_service: respx.MockRouter):
    """Callable fixture: register happy-path game + team stats mocks."""

    def _mock(game_id: str, status: str = "SCHEDULED") -> None:
        stats_service.get(f"/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=envelope(game_payload(game_id, status)))
        )
        stats_service.get("/api/v1/stats/teams/team-home/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(team_stats_payload("team-home", 116.0, 110.0)))
        )
        stats_service.get("/api/v1/stats/teams/team-away/stats", params={"stat_type": "all"}).mock(
            return_value=Response(200, json=envelope(team_stats_payload("team-away", 112.0, 113.0)))
        )

    return _mock


@pytest.fixture
def payloads():
    """Access to raw payload builders for tests that need custom mocks."""
    return {"envelope": envelope, "game": game_payload, "team_stats": team_stats_payload}
