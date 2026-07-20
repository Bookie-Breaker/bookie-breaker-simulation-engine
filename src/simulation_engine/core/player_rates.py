"""Map statistics-service player details to per-sport PlayerRates (Phase 7 Wave 3).

Stat keys are the CANONICAL cross-service keys — the Odds API market keys per
ADR-029 — so downstream services (prediction-engine, agent, emulator) can join
player distributions to market lines without translation:

- SOCCER: ``player_goal_scorer_anytime`` (YES_NO), ``player_shots``,
  ``player_shots_on_target`` (OVER_UNDER)
- BASKETBALL: ``player_points``, ``player_rebounds``, ``player_assists``,
  ``player_threes``, ``player_points_rebounds_assists`` (OVER_UNDER)
- BASEBALL: ``batter_hits``, ``batter_total_bases``, ``batter_home_runs``,
  ``pitcher_strikeouts`` — DORMANT in v1 (empty MLB rosters upstream)
- FOOTBALL: ``player_pass_yds``, ``player_rush_yds``, ``player_reception_yds``,
  ``player_receptions``, ``player_anytime_td`` — DORMANT in v1 (empty rosters)
- HOCKEY: none in v1.

Every derived rate below is an approximation from season aggregates; the
docstrings of the per-sport builders document each modeling choice.
"""

import logging
from typing import Literal

from simulation_engine.clients.statistics import PlayerDetail
from simulation_engine.core import league_averages as lg
from simulation_engine.core.params import PlayerRates

logger = logging.getLogger(__name__)

#: Stats settled YES/NO (count > 0) instead of over/under a line.
YES_NO_STAT_KEYS = frozenset({"player_goal_scorer_anytime", "player_anytime_td"})

SOCCER_STAT_KEYS = ("player_goal_scorer_anytime", "player_shots", "player_shots_on_target")
BASKETBALL_STAT_KEYS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
)

_SOCCER_MATCH_MINUTES = 90.0
_NBA_REGULATION_MINUTES = 48.0
#: Smoothing pseudo-goals so a rotation player with 0 season goals keeps a
#: small nonzero anytime-scorer probability instead of exactly zero.
_GOAL_SMOOTHING = 0.5
#: Share of NBA team points scored on three-pointers (league-wide, ~2025).
#: Used only for the estimated-threes heuristic below.
_NBA_POINTS_FROM_THREES_SHARE = 0.32
_SOCCER_GOALKEEPER_POSITIONS = frozenset({"G", "GK", "GOALKEEPER"})
_EXCLUDED_STATUSES = frozenset({"OUT"})


def _display_name(player: PlayerDetail) -> str:
    return f"{player.first_name} {player.last_name}".strip() or player.id


def _soccer_rates(players: list[PlayerDetail], team: Literal["HOME", "AWAY"]) -> list[PlayerRates]:
    """Soccer allocation rates from soccer_season_stats.

    Modeling choices (documented approximations):

    - Goal-share weights are ``(season goals + 0.5) x minutes_share`` where
      ``minutes_share = minutes / (appearances x 90)`` clipped to [0, 1] — the
      typical fraction of a match the player plays. Smoothing (+0.5) keeps
      low-sample players plausible; minutes weighting stops benchwarmers from
      inheriting the same pseudo-goal mass as starters. Shares are normalized
      within the team, so the multinomial goal allocation conserves team goals.
    - Shot / shots-on-target rates are per-90 rates scaled by the expected
      minutes fraction, which reduces algebraically to shots / appearances
      (and SOT / appearances).
    - Goalkeepers and OUT players are excluded entirely.
    """
    eligible = []  # (player, stats, minutes_share, weight)
    for player in players:
        if player.status.upper() in _EXCLUDED_STATUSES:
            continue
        if player.position.upper() in _SOCCER_GOALKEEPER_POSITIONS:
            continue
        stats = player.soccer_season_stats
        if stats is None or stats.appearances <= 0:
            continue
        minutes_share = min(stats.minutes / (stats.appearances * _SOCCER_MATCH_MINUTES), 1.0)
        weight = (stats.goals + _GOAL_SMOOTHING) * minutes_share
        eligible.append((player, stats, minutes_share, weight))

    total_weight = sum(weight for _, _, _, weight in eligible)
    if total_weight <= 0.0:
        return []

    rates: list[PlayerRates] = []
    for player, stats, minutes_share, weight in eligible:
        rates.append(
            PlayerRates(
                player_id=player.id,
                name=_display_name(player),
                position=player.position,
                team=team,
                rates={
                    "goal_share": weight / total_weight,
                    "shots_per_match": stats.shots / stats.appearances,
                    "sot_per_match": stats.shots_on_target / stats.appearances,
                    "minutes_share": minutes_share,
                },
            )
        )
    return rates


def _basketball_rates(players: list[PlayerDetail], team: Literal["HOME", "AWAY"]) -> list[PlayerRates]:
    """Basketball allocation rates from the basketball-shaped season_stats block.

    Modeling choices (documented approximations):

    - ``points_weight = points_per_game x minutes_share`` with
      ``minutes_share = minutes_per_game / 48`` clipped to [0, 1]. PPG already
      reflects playing time, so the extra minutes factor deliberately
      downweights low-minute players further (bench garbage-time scorers
      inflate PPG-only shares). The plugin normalizes weights within the team.
    - Rebounds/assists use the season per-game values directly as Poisson
      rates (scaled per iteration by the plugin's pace factor).
    - Threes: PlayerSeasonStats carries NO made-threes field (only
      three_point_pct), so 3PM/game is ESTIMATED as
      ``PPG x 0.32 / 3 x clip(player 3P% / league 3P%, 0, 1.8)`` — the
      league-wide share of points from threes, tilted by how good a shooter
      the player is, and zeroed for players with no recorded 3P% (bigs).
      This is the least-fabricated mapping available from the current
      contract; replace with real 3PM/game when the field exists.
    - OUT players are excluded.
    """
    rates: list[PlayerRates] = []
    for player in players:
        if player.status.upper() in _EXCLUDED_STATUSES:
            continue
        stats = player.season_stats
        if stats is None or stats.games_played <= 0:
            continue
        minutes_share = min(stats.minutes_per_game / _NBA_REGULATION_MINUTES, 1.0)
        shooter_tilt = (
            min(max(stats.three_point_pct / lg.NBA_THREE_PCT, 0.0), 1.8) if stats.three_point_pct > 0 else 0.0
        )
        threes_per_game = stats.points_per_game * _NBA_POINTS_FROM_THREES_SHARE / 3.0 * shooter_tilt
        rates.append(
            PlayerRates(
                player_id=player.id,
                name=_display_name(player),
                position=player.position,
                team=team,
                rates={
                    "points_weight": stats.points_per_game * minutes_share,
                    "rebounds_per_game": stats.rebounds_per_game,
                    "assists_per_game": stats.assists_per_game,
                    "threes_per_game": threes_per_game,
                    "minutes_share": minutes_share,
                },
            )
        )
    if not any(p.rates["points_weight"] > 0 for p in rates):
        return []
    return rates


def build_player_rates(
    plugin_label: str, players: list[PlayerDetail], team: Literal["HOME", "AWAY"]
) -> list[PlayerRates]:
    """Build PlayerRates for one team's roster, dispatched on the plugin label.

    Returns an empty list for dormant sports (baseball, football, hockey —
    their providers return empty rosters in v1 anyway) and for rosters with
    no usable season stats; callers proceed without player output and log.
    """
    if not players:
        return []
    if plugin_label == "soccer":
        return _soccer_rates(players, team)
    if plugin_label == "basketball":
        return _basketball_rates(players, team)
    logger.info(
        "player props are dormant for plugin %r until real roster data exists; ignoring %d players",
        plugin_label,
        len(players),
    )
    return []
