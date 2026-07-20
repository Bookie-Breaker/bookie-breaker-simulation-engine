"""Basketball team parameters and mapping from statistics-service responses.

Parameters the Phase 1 statistics-service contract does not expose
(three_attempt_rate, ft_rate, forced_tov_pct, opp_oreb_pct) fall back to NBA
league averages; two-point percentages are derived from the FG%/3P% mixture.
"""

from dataclasses import dataclass, field
from typing import Literal

from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core import league_averages as lg


@dataclass(frozen=True)
class SportParams:
    """Marker base for per-sport simulator parameter dataclasses.

    Shared plumbing (runner, hashing, PluginSpec) is typed against this base;
    each plugin narrows to its own subclass inside ``set_parameters``.
    """


@dataclass(frozen=True)
class TeamParams(SportParams):
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
class LiveState:
    """Current in-game state for live re-simulation (Phase 7 Wave 2).

    Plugins condition on this to simulate only the REMAINDER of the game and
    add the current score as a constant offset. ``fraction_remaining`` is the
    fraction of regulation game time still to be played, in (0, 1]. The
    sport-specific fields are optional refinements:

    - ``period`` / ``clock_seconds``: current period (quarter, inning, half,
      ...) and seconds remaining on the game clock. Carried for hashing and
      diagnostics by most plugins; baseball reads ``period`` as the inning
      number for its explicit-state resume.
    - ``bases`` / ``outs`` / ``half`` (baseball): base occupancy as a 3-char
      string (position i = runner on base i+1, "-" when empty: "---", "1--",
      "-2-", "--3", "12-", "1-3", "-23", "123"), outs in the current
      half-inning (0-2), and "TOP"/"BOTTOM".
    - ``possession`` / ``down`` / ``yardline`` (football): "HOME"/"AWAY" ball
      possession, current down (1-4), and yards from the opponent goal line
      (0-100). Only ``possession`` affects the drive-based model; down and
      yardline are carried for hashing and future play-level models.

    None-valued fields are stripped from the parameter hash (core/hashing.py)
    just like optional GameContext fields, so every distinct live state gets
    its own cache entry and omitted refinements do not fragment the cache.
    """

    home_score: int
    away_score: int
    fraction_remaining: float
    period: int | None = None
    clock_seconds: int | None = None
    bases: str | None = None
    outs: int | None = None
    half: str | None = None
    possession: str | None = None
    down: int | None = None
    yardline: int | None = None


@dataclass(frozen=True)
class PlayerRates:
    """Per-player allocation rates for the player-prop detailed path (Phase 7 Wave 3).

    ``rates`` keys are sport-specific and produced by ``core/player_rates.py``
    from statistics-service player season stats:

    - SOCCER: ``goal_share`` (normalized within the team, sums to 1.0),
      ``shots_per_match`` / ``sot_per_match`` (expected shots / shots on
      target for a typical appearance, i.e. per-90 rate x expected minutes
      fraction), ``minutes_share`` (typical fraction of a match played).
    - BASKETBALL: ``points_weight`` (unnormalized share weight: season PPG x
      minutes share; the plugin normalizes within the team),
      ``rebounds_per_game`` / ``assists_per_game`` / ``threes_per_game``
      (season per-game rates; threes are estimated, see player_rates.py),
      ``minutes_share`` (minutes per game / regulation minutes).
    - BASEBALL / FOOTBALL: reserved — their statistics providers return empty
      rosters in v1, so no rates are produced (plugins stay dormant).
    """

    player_id: str
    name: str
    position: str
    team: Literal["HOME", "AWAY"]
    rates: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GameContext:
    """Game-level context passed to plugins."""

    league: str = "NBA"
    neutral_site: bool = False
    # Probable starting pitchers (Phase 6 Wave 2, BASEBALL leagues only). FIP
    # is the only starter input the baseball plugin's multiplier needs; None
    # means unannounced. None-valued fields are stripped from the parameter
    # hash (core/hashing.py) so pre-baseball hashes stay byte-identical, and a
    # starter announcement naturally invalidates cached simulations.
    home_starter_fip: float | None = None
    away_starter_fip: float | None = None
    # Live re-simulation state (Phase 7 Wave 2). None means pregame. Because
    # None context fields are stripped from the parameter hash, pregame
    # hashes stay byte-identical to pre-Wave-2 hashes and every distinct live
    # state gets its own cache entry for free.
    live_state: LiveState | None = None
    # Player-prop roster identity (Phase 7 Wave 3). A stable SHA over the
    # sorted PlayerRates inputs (core/hashing.py compute_roster_signature),
    # set only when include_player_props is requested. None means props off —
    # the recursive None-strip keeps pregame hashes byte-identical to
    # pre-Wave-3 hashes, while any roster change (injury, transfer, rate
    # drift) naturally invalidates cached props simulations.
    roster_signature: str | None = None


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
