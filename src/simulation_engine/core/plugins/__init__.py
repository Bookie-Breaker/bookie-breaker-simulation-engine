"""Sport-specific simulation plugins and the per-league plugin registry."""

from collections.abc import Callable
from dataclasses import dataclass, field

from simulation_engine.api.errors import UnprocessableError
from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core.framework import GameSimulator
from simulation_engine.core.params import TeamParams, map_team_stats
from simulation_engine.core.plugins.basketball import BasketballSimulator
from simulation_engine.core.runner import BASKETBALL_GRID_CONFIG, GridConfig


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
    map_team_stats: Callable[[TeamStats], TeamParams]
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
