"""Poisson-grid hockey simulator with OT/shootout resolution (Phase 6 Wave 4).

Implements the hockey plugin from components/simulation-engine.md and ADR-026
by reusing the soccer plugin's joint-Poisson grid machinery (core/poisson_grid.py):

- **Goal rates.** ``lam_home = league_goals_per_team x (gf_home / league) x
  (ga_away / league) x home_mult`` (NHL: 3.0 goals/team, home mult 1.05, 1.0
  at neutral sites; symmetric for the away side without the multiplier). A
  special-teams adjustment then multiplies each side's rate by
  ``1 + PP_WEIGHT x (pp_pct_own - league_pp) - PK_WEIGHT x (pk_pct_opp - league_pk)``
  with small weights (0.5 / 0.5, league PP 21%, PK 79%; documented tunables in
  league_averages.py). Rates are clamped to [1.0, 6.0].
- **Regulation.** A 10x10 joint PMF grid over 0-9 goals per side (>0.999 of
  the Poisson mass at hockey rates; the shared grid renormalization absorbs
  the tail) with the Dixon-Coles low-score correction at rho = -0.05, sampled
  as one vectorized categorical draw per game.
- **OT/SO resolution.** Regulation ties are decided by a Bernoulli winner
  draw weighted by relative goal rates (``lam_home / (lam_home + lam_away)``),
  with ~50% of ties tagged as decided in overtime and the rest by shootout
  (same winner distribution; the split is a diagnostic, not a model change).
  The winner gets +1 goal on the FINAL score — the NHL convention where a
  shootout counts as one goal — so totals and moneylines settle on the final
  including OT/SO and the output has NO draws. Full 3-on-3 overtime dynamics
  are deliberately deferred to Phase 7+; regulation-time three-way markets
  would reuse the pre-resolution grid when wanted.
- **Live re-simulation (Phase 7 Wave 2).** With ``context.live_state`` set,
  the clipped pregame goal rates are scaled by ``fraction_remaining``
  (uniform-intensity approximation over regulation; the Dixon-Coles
  correction is reapplied to the remainder grid), the grid covers only the
  REGULATION time remaining, and the current score is added as a constant
  offset. OT/shootout resolution is unchanged and applies when the COMBINED
  score (current + sampled remainder) is tied at regulation end; the winner
  draw uses the live rates, whose ratio equals the pregame ratio. The
  pregame path (live_state=None) is bit-identical to pre-Wave-2 behavior.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core import league_averages as lg
from simulation_engine.core.framework import GameResult, GameSimulator
from simulation_engine.core.params import GameContext, SportParams
from simulation_engine.core.poisson_grid import build_goal_grid, sample_grid

_MAX_GOALS = 9
_GRID_SIZE = _MAX_GOALS + 1
_LAMBDA_MIN = 1.0
_LAMBDA_MAX = 6.0


@dataclass(frozen=True)
class HockeyParams(SportParams):
    """Hockey simulation parameters for one team.

    Goal rates feed the Poisson grid; special-teams percentages (fractions,
    e.g. 0.21) feed the small multiplicative adjustment. ``team_save_pct``
    is carried for hashing and debugging — goaltending is already baked into
    goals-against per game.
    """

    goals_for_per_game: float
    goals_against_per_game: float
    power_play_pct: float
    penalty_kill_pct: float
    team_save_pct: float


def map_hockey_stats(stats: TeamStats) -> HockeyParams:
    """Convert a statistics-service team stats response into hockey parameters.

    Empty hockey blocks fall back to NHL league averages, mirroring the other
    sport mappers.
    """
    hockey = stats.hockey
    return HockeyParams(
        goals_for_per_game=hockey.goals_for_per_game if hockey.goals_for_per_game > 0 else lg.NHL_GOALS_PER_TEAM,
        goals_against_per_game=(
            hockey.goals_against_per_game if hockey.goals_against_per_game > 0 else lg.NHL_GOALS_PER_TEAM
        ),
        power_play_pct=hockey.power_play_pct if hockey.power_play_pct > 0 else lg.NHL_LEAGUE_PP_PCT,
        penalty_kill_pct=hockey.penalty_kill_pct if hockey.penalty_kill_pct > 0 else lg.NHL_LEAGUE_PK_PCT,
        team_save_pct=hockey.team_save_pct if hockey.team_save_pct > 0 else lg.NHL_LEAGUE_SAVE_PCT,
    )


def _config_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value) if isinstance(value, int | float) else default


class HockeySimulator(GameSimulator):
    """Vectorized Poisson-grid NHL simulator with OT/SO tie resolution."""

    def __init__(self, plugin_config: dict[str, object] | None = None) -> None:
        config = plugin_config or {}
        self._league_goals = _config_float(config, "league_goals_per_team", lg.NHL_GOALS_PER_TEAM)
        self._home_goal_mult = _config_float(config, "home_goal_mult", lg.NHL_HOME_GOAL_MULT)
        self._pp_weight = _config_float(config, "pp_weight", lg.HOCKEY_PP_WEIGHT)
        self._pk_weight = _config_float(config, "pk_weight", lg.HOCKEY_PK_WEIGHT)
        self._league_pp = _config_float(config, "league_pp_pct", lg.NHL_LEAGUE_PP_PCT)
        self._league_pk = _config_float(config, "league_pk_pct", lg.NHL_LEAGUE_PK_PCT)
        self._dc_rho = _config_float(config, "dc_rho", lg.HOCKEY_DC_RHO)
        self._ot_share = _config_float(config, "ot_share_of_ties", lg.NHL_OT_SHARE_OF_TIES)
        league = config.get("league")
        self._league_override = league if isinstance(league, str) else None
        self._league = self._league_override or "NHL"
        self._lam_home = 0.0
        self._lam_away = 0.0
        self._offset_home = 0
        self._offset_away = 0
        self._grid: npt.NDArray[np.float64] | None = None
        self._cdf: npt.NDArray[np.float64] | None = None
        # Diagnostics from the most recent simulate_games call.
        self._last_regulation_tied: npt.NDArray[np.bool_] = np.zeros(0, dtype=np.bool_)
        self._last_ot_decided = 0
        self._last_so_decided = 0

    def _special_teams_multiplier(self, own: HockeyParams, opponent: HockeyParams) -> float:
        """Small multiplicative bump: strong own PP raises the rate, strong opposing PK lowers it."""
        return (
            1.0
            + self._pp_weight * (own.power_play_pct - self._league_pp)
            - self._pk_weight * (opponent.penalty_kill_pct - self._league_pk)
        )

    def set_parameters(self, home_params: SportParams, away_params: SportParams, context: GameContext) -> None:
        if not isinstance(home_params, HockeyParams) or not isinstance(away_params, HockeyParams):
            raise TypeError("HockeySimulator requires HockeyParams for both teams")
        if self._league_override is None:
            self._league = context.league

        home_mult = 1.0 if context.neutral_site else self._home_goal_mult
        lam_home = (
            self._league_goals
            * (home_params.goals_for_per_game / self._league_goals)
            * (away_params.goals_against_per_game / self._league_goals)
            * home_mult
            * self._special_teams_multiplier(home_params, away_params)
        )
        lam_away = (
            self._league_goals
            * (away_params.goals_for_per_game / self._league_goals)
            * (home_params.goals_against_per_game / self._league_goals)
            * self._special_teams_multiplier(away_params, home_params)
        )
        self._lam_home = float(np.clip(lam_home, _LAMBDA_MIN, _LAMBDA_MAX))
        self._lam_away = float(np.clip(lam_away, _LAMBDA_MIN, _LAMBDA_MAX))
        live = context.live_state
        if live is not None:
            # Remainder-of-regulation rates: scale the clipped full-game
            # lambdas by the fraction of regulation remaining. No
            # re-clamping — a small remainder legitimately falls below the
            # pregame lambda floor. The OT winner draw uses the ratio of
            # these rates, which the common factor leaves unchanged.
            self._lam_home *= live.fraction_remaining
            self._lam_away *= live.fraction_remaining
            self._offset_home = live.home_score
            self._offset_away = live.away_score
        else:
            self._offset_home = 0
            self._offset_away = 0
        self._grid = build_goal_grid(self._lam_home, self._lam_away, self._dc_rho, _GRID_SIZE)
        cdf = np.cumsum(self._grid.ravel())
        cdf[-1] = 1.0  # guard the inverse-CDF draw against float round-off
        self._cdf = cdf

    def _require_cdf(self) -> npt.NDArray[np.float64]:
        if self._cdf is None:
            raise RuntimeError("set_parameters must be called before simulating")
        return self._cdf

    def _sample_regulation(
        self, rng: np.random.Generator, n: int
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Regulation (60-minute) scores straight off the grid; ties are valid here."""
        return sample_grid(rng, self._require_cdf(), _GRID_SIZE, n)

    def simulate_games(self, rng: np.random.Generator, n: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        home_scores, away_scores = self._sample_regulation(rng, n)
        # Live runs: regulation final = current score + sampled remainder, so
        # the tie check below covers games tied at REGULATION END including
        # the current score. Adding zero offsets pregame changes nothing.
        home_scores = home_scores + np.int32(self._offset_home)
        away_scores = away_scores + np.int32(self._offset_away)

        tied = home_scores == away_scores
        self._last_regulation_tied = tied.copy()
        tied_idx = np.flatnonzero(tied)
        if tied_idx.size:
            # OT-vs-shootout tag first (diagnostic), then the winner draw, so
            # the RNG stream is deterministic and seed-stable.
            in_overtime = rng.random(tied_idx.size) < self._ot_share
            p_home = self._lam_home / (self._lam_home + self._lam_away)
            home_wins = rng.random(tied_idx.size) < p_home
            # NHL convention: OT and shootout wins both add exactly one goal
            # to the winner's FINAL score.
            home_scores[tied_idx[home_wins]] += 1
            away_scores[tied_idx[~home_wins]] += 1
            self._last_ot_decided = int(in_overtime.sum())
            self._last_so_decided = int(tied_idx.size - self._last_ot_decided)
        else:
            self._last_ot_decided = 0
            self._last_so_decided = 0

        return home_scores, away_scores

    def simulate_game(self, rng: np.random.Generator) -> GameResult:
        home, away = self.simulate_games(rng, 1)
        return GameResult(home_score=int(home[0]), away_score=int(away[0]), metadata={})

    def joint_grid(self) -> npt.NDArray[np.float64] | None:
        """The Dixon-Coles joint REGULATION-score PMF built by set_parameters.

        Hockey final scores add +1 to the OT/shootout winner, so this grid is
        the pre-resolution (60-minute) joint distribution — the right object
        for regulation-time three-way markets, not for final-score legs. For
        live runs the remainder grid is shifted by the current score
        (zero-padded low rows/columns) so it stays the joint PMF of
        regulation-end scores.
        """
        if self._grid is None:
            return None
        if self._offset_home == 0 and self._offset_away == 0:
            return self._grid
        shifted = np.zeros((_GRID_SIZE + self._offset_home, _GRID_SIZE + self._offset_away), dtype=np.float64)
        shifted[self._offset_home :, self._offset_away :] = self._grid
        return shifted

    def get_sport(self) -> str:
        return "HOCKEY"

    def get_league(self) -> str:
        return self._league
