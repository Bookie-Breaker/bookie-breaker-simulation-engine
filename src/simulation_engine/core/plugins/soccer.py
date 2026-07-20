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

Live re-simulation (Phase 7 Wave 2): with ``context.live_state`` set, the
pregame goal rates are scaled by ``fraction_remaining`` (a homogeneous-Poisson
approximation — scoring intensity is assumed uniform over the match, and the
Dixon-Coles correction is reapplied to the remainder grid's low-score cells),
the goal grid is rebuilt for the remaining match only, and the current score
is added as a constant offset to every sampled final score. The pregame path
(live_state=None) is bit-identical to pre-Wave-2 behavior.
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core import league_averages as lg
from simulation_engine.core.framework import BatchResult, GameResult, GameSimulator
from simulation_engine.core.params import GameContext, PlayerRates, SportParams
from simulation_engine.core.poisson_grid import build_goal_grid, poisson_pmf, sample_grid

logger = logging.getLogger(__name__)

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
        self._offset_home = 0
        self._offset_away = 0
        self._grid: npt.NDArray[np.float64] | None = None
        self._cdf: npt.NDArray[np.float64] | None = None
        self._players_home: list[PlayerRates] = []
        self._players_away: list[PlayerRates] = []

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
        live = context.live_state
        if live is not None:
            # Remainder-of-match rates: scale the clipped full-match lambdas
            # by the fraction of the match remaining (uniform-intensity
            # approximation). No re-clamping — a small remainder legitimately
            # falls below the pregame lambda floor.
            self._lam_home *= live.fraction_remaining
            self._lam_away *= live.fraction_remaining
            self._offset_home = live.home_score
            self._offset_away = live.away_score
        else:
            self._offset_home = 0
            self._offset_away = 0
        self._grid = _build_goal_grid(self._lam_home, self._lam_away, self._dc_rho)
        cdf = np.cumsum(self._grid.ravel())
        cdf[-1] = 1.0  # guard the inverse-CDF draw against float round-off
        self._cdf = cdf

    def _require_cdf(self) -> npt.NDArray[np.float64]:
        if self._cdf is None:
            raise RuntimeError("set_parameters must be called before simulating")
        return self._cdf

    def simulate_games(self, rng: np.random.Generator, n: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        home, away = sample_grid(rng, self._require_cdf(), _GRID_SIZE, n)
        # Live runs: final score = current score + sampled remainder. Adding
        # zero offsets pregame leaves values (and the RNG stream) unchanged.
        return home + np.int32(self._offset_home), away + np.int32(self._offset_away)

    def simulate_game(self, rng: np.random.Generator) -> GameResult:
        home, away = self.simulate_games(rng, 1)
        return GameResult(home_score=int(home[0]), away_score=int(away[0]), metadata={})

    def set_players(self, home: list[PlayerRates], away: list[PlayerRates]) -> None:
        """Store rosters for the detailed path (Phase 7 Wave 3); survives set_parameters."""
        self._players_home = list(home)
        self._players_away = list(away)

    def simulate_games_detailed(self, rng: np.random.Generator, n: int) -> BatchResult:
        """Team scores plus per-player goal/shot arrays (Phase 7 Wave 3).

        Player allocation model (documented approximations):

        - Each team's SIMULATED goals (the sampled remainder for live runs —
          goals already on the board cannot be attributed) are split across
          the roster with one multinomial draw per iteration over the
          precomputed ``goal_share`` vector, so player goals conserve team
          goals exactly and inherit the Dixon-Coles team correlation.
        - ``player_goal_scorer_anytime`` is the per-player goals array itself;
          the output layer settles it YES/NO as P(goals > 0).
        - Shots and shots-on-target are INDEPENDENT Poisson draws per
          iteration with per-player per-match rates (season shots per
          appearance = per-90 rate x expected minutes fraction). They are not
          coupled to each other or to goals, so team-level correlation flows
          only through the goal allocation — a player can sample more goals
          than shots on target in an iteration; marginal distributions are
          the modeling target, not within-iteration consistency.

        The pregame team path (``simulate_games``) is untouched; this method
        draws from the same grid, so team-level arrays here are statistically
        identical (not byte-identical across chunks, since player draws
        advance the shared RNG stream).
        """
        home_raw, away_raw = sample_grid(rng, self._require_cdf(), _GRID_SIZE, n)
        player_stats: dict[str, dict[str, npt.NDArray[np.int32]]] = {}
        for roster, team_goals in ((self._players_home, home_raw), (self._players_away, away_raw)):
            self._allocate_team(rng, roster, team_goals, player_stats)
        return BatchResult(
            home_scores=home_raw + np.int32(self._offset_home),
            away_scores=away_raw + np.int32(self._offset_away),
            player_stats=player_stats,
        )

    @staticmethod
    def _allocate_team(
        rng: np.random.Generator,
        roster: list[PlayerRates],
        team_goals: npt.NDArray[np.int32],
        player_stats: dict[str, dict[str, npt.NDArray[np.int32]]],
    ) -> None:
        if not roster:
            logger.info("soccer player allocation skipped: empty roster (no player output for this team)")
            return
        shares = np.array([player.rates.get("goal_share", 0.0) for player in roster], dtype=np.float64)
        total = shares.sum()
        if total <= 0.0:
            logger.info("soccer player allocation skipped: zero goal-share mass (no player output for this team)")
            return
        shares = shares / total  # guard float drift; player_rates normalizes already
        shot_rates = np.array([player.rates.get("shots_per_match", 0.0) for player in roster], dtype=np.float64)
        sot_rates = np.array([player.rates.get("sot_per_match", 0.0) for player in roster], dtype=np.float64)

        n = len(team_goals)
        goals = rng.multinomial(team_goals.astype(np.int64), shares).astype(np.int32)  # (n, k)
        shots = rng.poisson(np.broadcast_to(shot_rates, (n, len(roster)))).astype(np.int32)
        sot = rng.poisson(np.broadcast_to(sot_rates, (n, len(roster)))).astype(np.int32)
        for j, player in enumerate(roster):
            player_stats[player.player_id] = {
                "player_goal_scorer_anytime": goals[:, j],
                "player_shots": shots[:, j],
                "player_shots_on_target": sot[:, j],
            }

    def joint_grid(self) -> npt.NDArray[np.float64] | None:
        """The Dixon-Coles joint regulation-score PMF built by set_parameters.

        Soccer settles on regulation scores (ADR-027), so this grid IS the
        analytic joint distribution of the simulated outcomes. For live runs
        the remainder grid is shifted by the current score (zero-padded low
        rows/columns) so it stays the joint PMF of FINAL scores.
        """
        if self._grid is None:
            return None
        if self._offset_home == 0 and self._offset_away == 0:
            return self._grid
        shifted = np.zeros((_GRID_SIZE + self._offset_home, _GRID_SIZE + self._offset_away), dtype=np.float64)
        shifted[self._offset_home :, self._offset_away :] = self._grid
        return shifted

    def get_sport(self) -> str:
        return "SOCCER"

    def get_league(self) -> str:
        return self._league
