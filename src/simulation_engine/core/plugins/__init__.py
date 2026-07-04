"""Sport-specific simulation plugins."""

from simulation_engine.api.errors import UnprocessableError
from simulation_engine.core.framework import GameSimulator
from simulation_engine.core.plugins.basketball import BasketballSimulator

_PLUGINS: dict[str, type[BasketballSimulator]] = {
    "NBA": BasketballSimulator,
}


def get_simulator(league: str, plugin_config: dict[str, object] | None = None) -> GameSimulator:
    """Return a simulator for the league, or raise 422 for unsupported leagues."""
    plugin_cls = _PLUGINS.get(league.upper())
    if plugin_cls is None:
        raise UnprocessableError(f"League {league!r} is not supported for simulation in Phase 2 (NBA only)")
    return plugin_cls(plugin_config or {})
