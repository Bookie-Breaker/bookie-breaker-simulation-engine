"""Dixon-Coles-adjusted independent-Poisson soccer simulator (Phase 6 Wave 1).

Implements the soccer plugin from components/simulation-engine.md and ADR-026:
each team's goal rate is ``base_goals_per_team x attack x opponent defense``
(times ``home_goal_multiplier`` for the home side outside neutral venues), a
13x13 joint PMF grid over 0-12 goals per side (>0.9999 of the Poisson mass)
gets the Dixon-Coles tau correction on the four low-score cells, and games are
sampled as one vectorized categorical draw over the flattened grid.

No overtime or tie-break is simulated: draws are valid outcomes and scores are
regulation (90-minute) scores, which is what all Phase 6 soccer markets settle
on (ADR-027). One simulator serves every SOCCER league; competitions differ
only by plugin config (FIFA_WC vs EPL base rates and home multiplier).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core import league_averages as lg
from simulation_engine.core.framework import GameResult, GameSimulator
from simulation_engine.core.params import GameContext, SportParams
from simulation_engine.core.poisson_grid import build_goal_grid, poisson_pmf, sample_grid

_MAX_GOALS = 12
_GRID_SIZE = _MAX_GOALS + 1
_LAMBDA_MIN = 0.2
_LAMBDA_MAX = 4.5
_LEAGUE_AVERAGE_STRENGTH = 1.0


@dataclass(frozen=True)
class SoccerParams(SportParams):
    """Soccer simulation parameters for one team.

    ``attack``/``defense`` are multiplicative strengths relative to the
    competition average (1.0 = average) and are the primary model inputs; the
    raw per-match goal rates are carried for hashing and debugging.
    """

    attack: float
    defense: float
    goals_for_per_match: float
    goals_against_per_match: float


def map_soccer_stats(stats: TeamStats) -> SoccerParams:
    """Convert a statistics-service team stats response into soccer parameters.

    Empty soccer blocks fall back to league-average strength 1.0, mirroring
    how the basketball mapper falls back to NBA league averages.
    """
    soccer = stats.soccer
    return SoccerParams(
        attack=soccer.attack_strength if soccer.attack_strength > 0 else _LEAGUE_AVERAGE_STRENGTH,
        defense=soccer.defense_strength if soccer.defense_strength > 0 else _LEAGUE_AVERAGE_STRENGTH,
        goals_for_per_match=soccer.goals_for_per_match,
        goals_against_per_match=soccer.goals_against_per_match,
    )


def _poisson_pmf(lam: float) -> npt.NDArray[np.float64]:
    """Poisson PMF over 0..12 (soccer-sized shim over the shared helper)."""
    return poisson_pmf(lam, _GRID_SIZE)


def _build_goal_grid(lam_home: float, lam_away: float, rho: float) -> npt.NDArray[np.float64]:
    """13x13 Dixon-Coles joint score PMF (soccer-sized shim over the shared helper).

    The grid machinery moved to core/poisson_grid.py in Phase 6 Wave 4 so the
    hockey plugin can reuse it with its own grid size and rho.
    """
    return build_goal_grid(lam_home, lam_away, rho, _GRID_SIZE)


def _config_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value) if isinstance(value, int | float) else default


class SoccerSimulator(GameSimulator):
    """Vectorized Dixon-Coles Poisson simulator shared by all SOCCER leagues."""

    def __init__(self, plugin_config: dict[str, object] | None = None) -> None:
        config = plugin_config or {}
        self._base_goals = _config_float(config, "base_goals_per_team", lg.SOCCER_WC_BASE_GOALS_PER_TEAM)
        self._home_goal_multiplier = _config_float(config, "home_goal_multiplier", 1.0)
        self._dc_rho = _config_float(config, "dc_rho", lg.SOCCER_DC_RHO)
        league = config.get("league")
        self._league_override = league if isinstance(league, str) else None
        self._league = self._league_override or "FIFA_WC"
        self._lam_home = 0.0
        self._lam_away = 0.0
        self._grid: npt.NDArray[np.float64] | None = None
        self._cdf: npt.NDArray[np.float64] | None = None

    def set_parameters(self, home_params: SportParams, away_params: SportParams, context: GameContext) -> None:
        if not isinstance(home_params, SoccerParams) or not isinstance(away_params, SoccerParams):
            raise TypeError("SoccerSimulator requires SoccerParams for both teams")
        if self._league_override is None:
            self._league = context.league

        multiplier = 1.0 if context.neutral_site else self._home_goal_multiplier
        self._lam_home = float(
            np.clip(self._base_goals * home_params.attack * away_params.defense * multiplier, _LAMBDA_MIN, _LAMBDA_MAX)
        )
        self._lam_away = float(
            np.clip(self._base_goals * away_params.attack * home_params.defense, _LAMBDA_MIN, _LAMBDA_MAX)
        )
        self._grid = _build_goal_grid(self._lam_home, self._lam_away, self._dc_rho)
        cdf = np.cumsum(self._grid.ravel())
        cdf[-1] = 1.0  # guard the inverse-CDF draw against float round-off
        self._cdf = cdf

    def _require_cdf(self) -> npt.NDArray[np.float64]:
        if self._cdf is None:
            raise RuntimeError("set_parameters must be called before simulating")
        return self._cdf

    def simulate_games(self, rng: np.random.Generator, n: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        return sample_grid(rng, self._require_cdf(), _GRID_SIZE, n)

    def simulate_game(self, rng: np.random.Generator) -> GameResult:
        home, away = self.simulate_games(rng, 1)
        return GameResult(home_score=int(home[0]), away_score=int(away[0]), metadata={})

    def get_sport(self) -> str:
        return "SOCCER"

    def get_league(self) -> str:
        return self._league
