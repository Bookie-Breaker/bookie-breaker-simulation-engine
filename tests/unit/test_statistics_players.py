"""StatisticsClient player-surface tests (Phase 7 Wave 3): list pagination + detail parsing."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from simulation_engine.api.errors import DependencyError, NotFoundError
from simulation_engine.clients.statistics import StatisticsClient

BASE = "http://stats.test"


@pytest.fixture
async def client() -> AsyncIterator[StatisticsClient]:
    async with httpx.AsyncClient() as http:
        yield StatisticsClient(BASE, http)


def player_summary(player_id: str, position: str = "F") -> dict[str, Any]:
    return {
        "id": player_id,
        "team_id": "team-1",
        "first_name": "First",
        "last_name": player_id.upper(),
        "position": position,
        "status": "ACTIVE",
    }


def list_envelope(items: list[dict[str, Any]], has_more: bool = False, next_cursor: str = "") -> dict[str, Any]:
    return {
        "data": items,
        "meta": {
            "timestamp": "2026-07-19T12:00:00Z",
            "request_id": "req-test",
            "pagination": {"limit": 200, "has_more": has_more, "next_cursor": next_cursor},
        },
    }


class TestGetTeamPlayers:
    @respx.mock
    async def test_single_page(self, client: StatisticsClient) -> None:
        respx.get(f"{BASE}/api/v1/stats/players", params={"league": "EPL", "team_id": "team-1", "limit": "200"}).mock(
            return_value=httpx.Response(200, json=list_envelope([player_summary("p1"), player_summary("p2")]))
        )
        players = await client.get_team_players("EPL", "team-1")
        assert [p.id for p in players] == ["p1", "p2"]
        assert players[0].position == "F"

    @respx.mock
    async def test_follows_cursor_pagination(self, client: StatisticsClient) -> None:
        route = respx.get(f"{BASE}/api/v1/stats/players")
        route.side_effect = [
            httpx.Response(200, json=list_envelope([player_summary("p1")], True, "cur-2")),
            httpx.Response(200, json=list_envelope([player_summary("p2")])),
        ]
        players = await client.get_team_players("EPL", "team-1")
        assert [p.id for p in players] == ["p1", "p2"]
        assert route.call_count == 2
        assert "cursor=cur-2" in str(route.calls[1].request.url)

    @respx.mock
    async def test_empty_roster_returns_empty_list(self, client: StatisticsClient) -> None:
        respx.get(f"{BASE}/api/v1/stats/players").mock(return_value=httpx.Response(200, json=list_envelope([])))
        assert await client.get_team_players("MLB", "team-1") == []

    @respx.mock
    async def test_malformed_envelope_raises_dependency_error(self, client: StatisticsClient) -> None:
        respx.get(f"{BASE}/api/v1/stats/players").mock(
            return_value=httpx.Response(200, json={"data": {"not": "a list"}})
        )
        with pytest.raises(DependencyError):
            await client.get_team_players("EPL", "team-1")


class TestGetPlayer:
    @respx.mock
    async def test_parses_soccer_season_stats(self, client: StatisticsClient) -> None:
        detail = {
            **player_summary("p1"),
            "soccer_season_stats": {
                "appearances": 20,
                "minutes": 1750,
                "goals": 12,
                "assists": 4,
                "shots": 55,
                "shots_on_target": 28,
                "yellow_cards": 2,
                "red_cards": 0,
            },
        }
        respx.get(f"{BASE}/api/v1/stats/players/p1").mock(
            return_value=httpx.Response(200, json={"data": detail, "meta": {}})
        )
        player = await client.get_player("p1")
        assert player.soccer_season_stats is not None
        assert player.soccer_season_stats.goals == 12
        assert player.season_stats is None

    @respx.mock
    async def test_parses_basketball_season_stats(self, client: StatisticsClient) -> None:
        detail = {
            **player_summary("p2", position="G"),
            "season_stats": {
                "season": 2026,
                "games_played": 62,
                "minutes_per_game": 35.1,
                "points_per_game": 27.4,
                "rebounds_per_game": 7.2,
                "assists_per_game": 8.1,
                "three_point_pct": 0.39,
            },
        }
        respx.get(f"{BASE}/api/v1/stats/players/p2").mock(
            return_value=httpx.Response(200, json={"data": detail, "meta": {}})
        )
        player = await client.get_player("p2")
        assert player.season_stats is not None
        assert player.season_stats.points_per_game == 27.4
        assert player.soccer_season_stats is None

    @respx.mock
    async def test_missing_player_raises_not_found(self, client: StatisticsClient) -> None:
        respx.get(f"{BASE}/api/v1/stats/players/ghost").mock(return_value=httpx.Response(404, json={}))
        with pytest.raises(NotFoundError):
            await client.get_player("ghost")
