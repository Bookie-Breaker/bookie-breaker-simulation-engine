"""Basketball team parameters and mapping from statistics-service responses.

Parameters the Phase 1 statistics-service contract does not expose
(three_attempt_rate, ft_rate, forced_tov_pct, opp_oreb_pct) fall back to NBA
league averages; two-point percentages are derived from the FG%/3P% mixture.
"""

from dataclasses import dataclass

from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core import league_averages as lg


@dataclass(frozen=True)
class TeamParams:
    """Basketball simulation parameters for one team (percentages as fractions)."""

    team_id: str
    abbreviation: str
    pace: float  # possessions per 48 minutes
    off_rating: float  # points per 100 possessions
    def_rating: float  # points allowed per 100 possessions
    three_pct: float
    two_pct: float
    ft_pct: float
    three_attempt_rate: float  # 3PA / FGA
    ft_rate: float  # FTA / FGA
    tov_pct: float  # turnovers per 100 possessions (percent, e.g. 13.0)
    oreb_pct: float  # offensive rebound percent (e.g. 27.0)
    opp_three_pct: float
    opp_two_pct: float
    opp_ft_rate: float
    forced_tov_pct: float
    opp_oreb_pct: float


@dataclass(frozen=True)
class GameContext:
    """Game-level context passed to plugins."""

    league: str = "NBA"
    neutral_site: bool = False


def _derive_two_pct(fg_pct: float, three_pct: float, three_attempt_rate: float) -> float:
    """Solve FG% = r * 3P% + (1 - r) * 2P% for the two-point percentage."""
    r = three_attempt_rate
    two_pct = (fg_pct - r * three_pct) / (1.0 - r)
    return min(max(two_pct, 0.30), 0.70)


def map_team_stats(stats: TeamStats) -> TeamParams:
    """Convert a statistics-service team stats response into plugin parameters."""
    off = stats.offensive
    dfn = stats.defensive
    adv = stats.advanced

    three_attempt_rate = lg.NBA_THREE_ATTEMPT_RATE
    three_pct = off.three_point_pct if off.three_point_pct > 0 else lg.NBA_THREE_PCT
    ft_pct = off.free_throw_pct if off.free_throw_pct > 0 else lg.NBA_FT_PCT

    return TeamParams(
        team_id=stats.team_id,
        abbreviation=stats.team_abbreviation,
        pace=off.pace if off.pace > 0 else lg.NBA_LEAGUE_AVG_PACE,
        off_rating=off.offensive_rating,
        def_rating=dfn.defensive_rating,
        three_pct=three_pct,
        two_pct=_derive_two_pct(off.field_goal_pct, three_pct, three_attempt_rate),
        ft_pct=ft_pct,
        three_attempt_rate=three_attempt_rate,
        ft_rate=lg.NBA_FT_RATE,
        tov_pct=adv.turnover_pct if adv.turnover_pct > 0 else lg.NBA_TOV_PCT,
        oreb_pct=adv.offensive_rebound_pct if adv.offensive_rebound_pct > 0 else lg.NBA_OREB_PCT,
        opp_three_pct=dfn.opponent_three_point_pct if dfn.opponent_three_point_pct > 0 else lg.NBA_THREE_PCT,
        opp_two_pct=_derive_two_pct(
            dfn.opponent_fg_pct, dfn.opponent_three_point_pct or lg.NBA_THREE_PCT, three_attempt_rate
        ),
        opp_ft_rate=lg.NBA_FT_RATE,
        forced_tov_pct=lg.NBA_TOV_PCT,
        opp_oreb_pct=lg.NBA_OREB_PCT,
    )
