"""Typed async client for the statistics-service REST API (port 8002)."""

import httpx
from pydantic import BaseModel, ConfigDict

from simulation_engine.api.errors import DependencyError, DependencyTimeoutError, NotFoundError


class TeamRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str = ""
    abbreviation: str = ""


class ProbablePitcher(BaseModel):
    """Probable starting pitcher for a game (BASEBALL leagues; present once
    announced, absent otherwise). Season pitching stats are embedded in the
    statistics-service payload so no extra lookup is needed."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    external_id: str = ""
    throws: str = ""  # throwing hand, L or R
    era: float = 0.0
    fip: float = 0.0
    k_bb_pct: float = 0.0  # strikeout rate minus walk rate
    innings_pitched: float = 0.0


class Game(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    league: str
    status: str
    home_team: TeamRef
    away_team: TeamRef
    scheduled_start: str = ""
    home_probable_pitcher: ProbablePitcher | None = None
    away_probable_pitcher: ProbablePitcher | None = None


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


class SoccerStats(BaseModel):
    """Soccer-specific stat block (SOCCER-sport leagues only; ADR-026).

    Strengths are multiplicative factors relative to the competition average
    (1.0 = average), shrunk toward 1.0 by matches played to damp small samples.
    """

    model_config = ConfigDict(extra="ignore")

    goals_for_per_match: float = 0.0
    goals_against_per_match: float = 0.0
    attack_strength: float = 0.0
    defense_strength: float = 0.0
    draws: int = 0
    form_goals_for_last5: float = 0.0
    form_goals_against_last5: float = 0.0
    form_points_last5: int = 0


class BaseballStats(BaseModel):
    """Baseball-specific stat block (BASEBALL-sport leagues only; ADR-026).

    FIP and wOBA are computed in-service from official counting stats using
    published seasonal constants.
    """

    model_config = ConfigDict(extra="ignore")

    runs_scored_per_game: float = 0.0
    runs_allowed_per_game: float = 0.0
    team_woba: float = 0.0
    team_obp: float = 0.0
    team_slg: float = 0.0
    batting_strikeout_pct: float = 0.0
    batting_walk_pct: float = 0.0
    team_era: float = 0.0
    team_fip: float = 0.0
    bullpen_era: float = 0.0


class FootballStats(BaseModel):
    """Football-specific stat block (FOOTBALL-sport leagues only; ADR-026).

    EPA metrics come from nflverse team stats (NFL) and are absent for
    NCAA_FB, which carries SP+ ratings from CFBD instead.
    """

    model_config = ConfigDict(extra="ignore")

    points_per_game: float = 0.0
    points_allowed_per_game: float = 0.0
    drives_per_game: float = 0.0
    points_per_drive_off: float = 0.0
    points_per_drive_def: float = 0.0
    epa_per_play_off: float = 0.0
    epa_per_play_def: float = 0.0
    turnover_margin_per_game: float = 0.0
    sp_plus_rating: float = 0.0


class HockeyStats(BaseModel):
    """Hockey-specific stat block (HOCKEY-sport leagues only; ADR-026)."""

    model_config = ConfigDict(extra="ignore")

    goals_for_per_game: float = 0.0
    goals_against_per_game: float = 0.0
    shots_for_per_game: float = 0.0
    shots_against_per_game: float = 0.0
    power_play_pct: float = 0.0
    penalty_kill_pct: float = 0.0
    team_save_pct: float = 0.0


class StatBlocks(BaseModel):
    model_config = ConfigDict(extra="ignore")

    offensive: OffensiveStats = OffensiveStats()
    defensive: DefensiveStats = DefensiveStats()
    advanced: AdvancedStats = AdvancedStats()
    soccer: SoccerStats = SoccerStats()
    baseball: BaseballStats = BaseballStats()
    football: FootballStats = FootballStats()
    hockey: HockeyStats = HockeyStats()


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
    soccer: SoccerStats = SoccerStats()
    baseball: BaseballStats = BaseballStats()
    football: FootballStats = FootballStats()
    hockey: HockeyStats = HockeyStats()


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
            soccer=parsed.stats.soccer,
            baseball=parsed.stats.baseball,
            football=parsed.stats.football,
            hockey=parsed.stats.hockey,
        )

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/v1/stats/health", timeout=1.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
