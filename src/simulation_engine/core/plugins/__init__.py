"""Sport-specific simulation plugins and the per-league plugin registry."""

from collections.abc import Callable
from dataclasses import dataclass, field

from simulation_engine.api.errors import UnprocessableError
from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core import league_averages as lg
from simulation_engine.core.framework import GameSimulator
from simulation_engine.core.params import SportParams, map_team_stats
from simulation_engine.core.plugins.baseball import BaseballSimulator, map_baseball_stats
from simulation_engine.core.plugins.basketball import BasketballSimulator
from simulation_engine.core.plugins.football import FootballSimulator, map_football_stats
from simulation_engine.core.plugins.hockey import HockeySimulator, map_hockey_stats
from simulation_engine.core.plugins.soccer import SoccerSimulator, map_soccer_stats
from simulation_engine.core.runner import BASKETBALL_GRID_CONFIG, GridConfig

#: Soccer's narrow line grids — goals, not points (components/simulation-engine.md).
SOCCER_GRID_CONFIG = GridConfig(spread_radius=3, total_radius=4)

#: Baseball line grids — the run line +/-1.5 dominates spreads; totals span wider.
BASEBALL_GRID_CONFIG = GridConfig(spread_radius=4, total_radius=6)

#: Football line grids — key numbers 3/7 demand wide spread coverage around the mean.
FOOTBALL_GRID_CONFIG = GridConfig(spread_radius=14, total_radius=16)

#: Hockey line grids — goals, like soccer (puck line +/-1.5 dominates spreads).
HOCKEY_GRID_CONFIG = GridConfig(spread_radius=3, total_radius=4)


@dataclass(frozen=True)
class PluginSpec:
    """Everything the service needs to simulate one league.

    Attributes:
        label: Engine identity used in cache-key hashing (e.g. "basketball").
            Shared by leagues that reuse the same simulator math.
        simulator: GameSimulator subclass, constructed with a plugin config dict.
        map_team_stats: Converts a statistics-service TeamStats response into
            the plugin's parameters object.
        grid_config: Per-sport spread/total line-grid radii for the runner.
        plugin_config: Default plugin configuration; request-level
            plugin_config entries override it. May be empty.
    """

    label: str
    simulator: type[GameSimulator]
    map_team_stats: Callable[[TeamStats], SportParams]
    grid_config: GridConfig
    plugin_config: dict[str, object] = field(default_factory=dict)


_PLUGINS: dict[str, PluginSpec] = {
    "NBA": PluginSpec(
        label="basketball",
        simulator=BasketballSimulator,
        map_team_stats=map_team_stats,
        grid_config=BASKETBALL_GRID_CONFIG,
        plugin_config={},
    ),
    # One soccer simulator serves every SOCCER competition (ADR-026); leagues
    # differ only by configuration. FIFA_WC plays at neutral venues, so its
    # home multiplier is 1.0.
    "FIFA_WC": PluginSpec(
        label="soccer",
        simulator=SoccerSimulator,
        map_team_stats=map_soccer_stats,
        grid_config=SOCCER_GRID_CONFIG,
        plugin_config={
            "base_goals_per_team": lg.SOCCER_WC_BASE_GOALS_PER_TEAM,
            "home_goal_multiplier": 1.0,
            "dc_rho": lg.SOCCER_DC_RHO,
        },
    ),
    "EPL": PluginSpec(
        label="soccer",
        simulator=SoccerSimulator,
        map_team_stats=map_soccer_stats,
        grid_config=SOCCER_GRID_CONFIG,
        plugin_config={
            "base_goals_per_team": lg.SOCCER_EPL_BASE_GOALS_PER_TEAM,
            "home_goal_multiplier": lg.SOCCER_EPL_HOME_GOAL_MULTIPLIER,
            "dc_rho": lg.SOCCER_DC_RHO,
        },
    ),
    # One baseball simulator serves MLB and NCAA_BSB (Phase 6 Wave 2);
    # college baseball differs only by its higher league scoring rate.
    "MLB": PluginSpec(
        label="baseball",
        simulator=BaseballSimulator,
        map_team_stats=map_baseball_stats,
        grid_config=BASEBALL_GRID_CONFIG,
        plugin_config={"league_runs_per_game": lg.MLB_RUNS_PER_GAME},
    ),
    "NCAA_BSB": PluginSpec(
        label="baseball",
        simulator=BaseballSimulator,
        map_team_stats=map_baseball_stats,
        grid_config=BASEBALL_GRID_CONFIG,
        plugin_config={"league_runs_per_game": lg.NCAA_BSB_RUNS_PER_GAME},
    ),
    # One drive-based football simulator serves NFL and NCAA_FB (Phase 6
    # Wave 3, ADR-018); college plays faster, scores more per drive, carries
    # a bigger home edge, and its overtime format never leaves ties.
    "NFL": PluginSpec(
        label="football",
        simulator=FootballSimulator,
        map_team_stats=map_football_stats,
        grid_config=FOOTBALL_GRID_CONFIG,
        plugin_config={
            "drives_mu": lg.NFL_DRIVES_PER_TEAM_MU,
            "drives_sigma": lg.NFL_DRIVES_SIGMA,
            "drives_clip_min": lg.NFL_DRIVES_CLIP_MIN,
            "drives_clip_max": lg.NFL_DRIVES_CLIP_MAX,
            "league_points_per_drive": lg.NFL_POINTS_PER_DRIVE,
            "hfa_margin_points": lg.NFL_HFA_MARGIN_POINTS,
            "ot_ties_allowed": True,
        },
    ),
    "NCAA_FB": PluginSpec(
        label="football",
        simulator=FootballSimulator,
        map_team_stats=map_football_stats,
        grid_config=FOOTBALL_GRID_CONFIG,
        plugin_config={
            "drives_mu": lg.NCAA_FB_DRIVES_PER_TEAM_MU,
            "drives_sigma": lg.NCAA_FB_DRIVES_SIGMA,
            "drives_clip_min": lg.NCAA_FB_DRIVES_CLIP_MIN,
            "drives_clip_max": lg.NCAA_FB_DRIVES_CLIP_MAX,
            "league_points_per_drive": lg.NCAA_FB_POINTS_PER_DRIVE,
            "hfa_margin_points": lg.NCAA_FB_HFA_MARGIN_POINTS,
            "ot_ties_allowed": False,
        },
    ),
    # Hockey (Phase 6 Wave 4). NHL only — NCAA_HKY stays gated per ADR-026.
    "NHL": PluginSpec(
        label="hockey",
        simulator=HockeySimulator,
        map_team_stats=map_hockey_stats,
        grid_config=HOCKEY_GRID_CONFIG,
        plugin_config={
            "league_goals_per_team": lg.NHL_GOALS_PER_TEAM,
            "home_goal_mult": lg.NHL_HOME_GOAL_MULT,
            "pp_weight": lg.HOCKEY_PP_WEIGHT,
            "pk_weight": lg.HOCKEY_PK_WEIGHT,
            "league_pp_pct": lg.NHL_LEAGUE_PP_PCT,
            "league_pk_pct": lg.NHL_LEAGUE_PK_PCT,
            "dc_rho": lg.HOCKEY_DC_RHO,
        },
    ),
    # NCAA basketball (Phase 6 Wave 5): the existing basketball simulator
    # with college configuration only — 40-minute game, ~68 possessions,
    # noisier pace, stronger home crowds, tighter possession clip bounds.
    "NCAA_BB": PluginSpec(
        label="basketball",
        simulator=BasketballSimulator,
        map_team_stats=map_team_stats,
        grid_config=BASKETBALL_GRID_CONFIG,
        plugin_config={
            "home_advantage": lg.NCAA_BB_HOME_ADVANTAGE,
            "league_avg_pace": lg.NCAA_BB_LEAGUE_AVG_PACE,
            "possession_std": lg.NCAA_BB_POSSESSION_STD,
            "possession_clip_min": lg.NCAA_BB_POSSESSION_CLIP_MIN,
            "possession_clip_max": lg.NCAA_BB_POSSESSION_CLIP_MAX,
            "ot_possession_fraction": lg.NCAA_BB_OT_POSSESSION_FRACTION,
        },
    ),
}


def get_plugin(league: str) -> PluginSpec:
    """Return the PluginSpec for the league, or raise 422 for unsupported leagues."""
    spec = _PLUGINS.get(league.upper())
    if spec is None:
        supported = ", ".join(sorted(_PLUGINS))
        raise UnprocessableError(f"League {league!r} is not supported for simulation (supported: {supported})")
    return spec


def get_simulator(league: str, plugin_config: dict[str, object] | None = None) -> GameSimulator:
    """Return a configured simulator for the league (convenience shim over get_plugin)."""
    spec = get_plugin(league)
    return spec.simulator({**spec.plugin_config, **(plugin_config or {})})
