"""Sport-agnostic simulation framework per algorithms/simulation-algorithms.md.

The GameSimulator ABC keeps the doc's single-game contract; simulate_games is
a batch hook plugins override with a vectorized implementation so the Monte
Carlo runner stays fast without changing the semantic contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from simulation_engine.core.params import GameContext, SportParams


@dataclass
class GameResult:
    """Result of a single simulated game."""

    home_score: int
    away_score: int
    metadata: dict[str, Any] = field(default_factory=dict)


class GameSimulator(ABC):
    """Interface that all sport-specific simulation plugins implement."""

    def __init__(self, plugin_config: dict[str, object] | None = None) -> None:
        """Registry construction contract: plugins accept an optional plugin-config dict."""
        self._plugin_config: dict[str, object] = plugin_config or {}

    @abstractmethod
    def set_parameters(self, home_params: SportParams, away_params: SportParams, context: GameContext) -> None:
        """Load team parameters (from statistics-service data) and game context.

        Plugins receive the params their PluginSpec mapper produced and narrow
        to their own SportParams subclass.
        """

    @abstractmethod
    def simulate_game(self, rng: np.random.Generator) -> GameResult:
        """Simulate one complete game. Must be deterministic given the rng state."""

    def simulate_games(self, rng: np.random.Generator, n: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Simulate n games, returning (home_scores, away_scores) arrays.

        Default implementation loops simulate_game; plugins override with a
        vectorized implementation.
        """
        home = np.zeros(n, dtype=np.int32)
        away = np.zeros(n, dtype=np.int32)
        for i in range(n):
            result = self.simulate_game(rng)
            home[i] = result.home_score
            away[i] = result.away_score
        return home, away

    def joint_grid(self) -> npt.NDArray[np.float64] | None:
        """Analytic joint score PMF (rows = home score, cols = away score) when the plugin has one.

        Poisson-grid sports (soccer, hockey) return the grid built by
        set_parameters; sports without an analytic joint return None (the
        default). Callers must invoke set_parameters first.
        """
        return None

    @abstractmethod
    def get_sport(self) -> str:
        """Return sport identifier: 'FOOTBALL', 'BASKETBALL', 'BASEBALL', 'SOCCER', or 'HOCKEY'."""

    @abstractmethod
    def get_league(self) -> str:
        """Return league identifier: 'NFL', 'NCAA_FB', 'NBA', 'NCAA_BB', 'MLB', 'NCAA_BSB',
        'FIFA_WC', 'EPL', 'NHL', or 'NCAA_HKY' (per ADR-026).
        """
