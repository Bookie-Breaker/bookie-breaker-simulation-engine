"""Typed async client for the statistics-service REST API (port 8002)."""

import httpx
from pydantic import BaseModel, ConfigDict

from simulation_engine.api.errors import DependencyError, DependencyTimeoutError, NotFoundError


class TeamRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str = ""
    abbreviation: str = ""


class Game(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    league: str
    status: str
    home_team: TeamRef
    away_team: TeamRef
    scheduled_start: str = ""


class OffensiveStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    points_per_game: float = 0.0
    field_goal_pct: float = 0.0
    three_point_pct: float = 0.0
    free_throw_pct: float = 0.0
    turnovers_per_game: float = 0.0
    offensive_rating: float = 0.0
    pace: float = 0.0
    effective_fg_pct: float = 0.0


class DefensiveStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    points_allowed_per_game: float = 0.0
    opponent_fg_pct: float = 0.0
    opponent_three_point_pct: float = 0.0
    defensive_rating: float = 0.0


class AdvancedStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    net_rating: float = 0.0
    true_shooting_pct: float = 0.0
    turnover_pct: float = 0.0
    offensive_rebound_pct: float = 0.0


class StatBlocks(BaseModel):
    model_config = ConfigDict(extra="ignore")

    offensive: OffensiveStats = OffensiveStats()
    defensive: DefensiveStats = DefensiveStats()
    advanced: AdvancedStats = AdvancedStats()


class TeamStatsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    team_id: str
    team_abbreviation: str = ""
    season: int = 0
    stats: StatBlocks = StatBlocks()


class TeamStats(BaseModel):
    """Flattened view used by the parameter mapper."""

    team_id: str
    team_abbreviation: str
    offensive: OffensiveStats
    defensive: DefensiveStats
    advanced: AdvancedStats


class StatisticsClient:
    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def _get(self, path: str, resource: str) -> dict[str, object]:
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.get(url)
        except httpx.TimeoutException as exc:
            raise DependencyTimeoutError(f"statistics-service timed out fetching {resource}") from exc
        except httpx.HTTPError as exc:
            raise DependencyError(f"statistics-service is unavailable: {exc}") from exc
        if response.status_code == 404:
            raise NotFoundError(f"{resource} not found in statistics-service")
        if response.status_code >= 500:
            raise DependencyError(f"statistics-service returned {response.status_code} for {resource}")
        payload: dict[str, object] = response.json()
        data = payload.get("data")
        if not isinstance(data, dict):
            raise DependencyError(f"statistics-service returned a malformed envelope for {resource}")
        return data

    async def get_game(self, game_id: str) -> Game:
        data = await self._get(f"/api/v1/stats/games/{game_id}", f"game {game_id}")
        return Game.model_validate(data)

    async def get_team_stats(self, team_id: str) -> TeamStats:
        data = await self._get(f"/api/v1/stats/teams/{team_id}/stats?stat_type=all", f"team stats {team_id}")
        parsed = TeamStatsResponse.model_validate(data)
        return TeamStats(
            team_id=parsed.team_id,
            team_abbreviation=parsed.team_abbreviation,
            offensive=parsed.stats.offensive,
            defensive=parsed.stats.defensive,
            advanced=parsed.stats.advanced,
        )

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/v1/stats/health", timeout=1.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
