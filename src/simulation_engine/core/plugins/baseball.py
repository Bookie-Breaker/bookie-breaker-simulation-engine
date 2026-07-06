"""Half-inning runs baseball simulator (Phase 6 Wave 2).

Implements the baseball plugin from components/simulation-engine.md following
ADR-018's granularity reasoning: half-inning run totals — not plate
appearances, which Phase 7's player props will need — are the coarsest unit
that still yields correct score, margin, and total shapes for the Phase 6 bet
types.

Model:

- **Half-inning run distribution.** Runs per half-inning (0-10) follow a
  zero-modified geometric distribution: ``P(0) = p0`` and
  ``P(k) = (1 - p0) * (1 - q) * q^(k-1)`` for ``k >= 1``, with the geometric
  tail truncated at 10 and renormalized so ``P(0)`` stays exactly ``p0``.
  The distribution is parameterized by ``q`` alone: the zero-inflation is
  tied to the target scoring rate as
  ``p0 = clip(P0_BASE - P0_ALPHA * (target - league_mean), 0.55, 0.85)``
  (``P0_BASE = 0.73``, ``P0_ALPHA = 0.35``; league_averages.py), then the
  shared bisection calibrator solves for ``q`` so the PMF mean matches the
  target to well under 1e-6. With ``P0_ALPHA = 0.35`` the untruncated
  variance-to-mean ratio ``(p0 + q) / (1 - q)`` stays above 1 (over-dispersed,
  like real innings) across the whole calibrated target range [0.2, 1.2].
- **Expected runs.** The batting team's expected runs per half-inning is the
  odds-ratio blend ``(team RS/G x opponent RA/G / league RS/G) / 9``, clamped
  to [0.2, 1.2] after multipliers.
- **Starter multiplier.** For innings 1-6 the batting team's rate is scaled
  by ``clip(opposing starter FIP / league FIP, 0.6, 1.6)`` when the opposing
  probable starter is announced (optional GameContext fields), else 1.0.
  Innings 7 onward use ``clip(opposing bullpen ERA / league ERA, 0.7, 1.5)``,
  falling back to the opposing team ERA, else a 1.0 multiplier. The four
  distinct distributions (home/away batting x starter/bullpen phase) are
  precomputed in ``set_parameters``.
- **Game flow.** Nine innings are drawn vectorized (18 categorical draws per
  chunk via searchsorted); the bottom of the 9th is zeroed where the home
  team already leads after 8 1/2 (a known ~0.2-0.4-run totals bias if
  omitted); tied games draw full extra innings at bullpen-phase rates until
  decided, so no draws appear in the output. A safety cap at 30 innings
  forces a coin-flip run; it never triggers at realistic parameters (tests
  assert this via ``_last_forced_tiebreaks``).

Documented approximations: walk-off innings are not truncated mid-inning
(home 9th/extra-inning runs count in full), no extra-innings ghost runner,
park factors are out of scope (no venue mapping source), and home advantage
enters only through the bats-last structure.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core import league_averages as lg
from simulation_engine.core.calibrate import calibrate_distribution
from simulation_engine.core.framework import GameResult, GameSimulator
from simulation_engine.core.params import GameContext, SportParams

_MAX_RUNS_PER_HALF_INNING = 10
_SUPPORT = _MAX_RUNS_PER_HALF_INNING + 1
_STARTER_INNINGS = 6  # innings 1-6 face the starter, 7+ the bullpen
_REGULATION_INNINGS = 9
_MAX_INNINGS = 30  # safety cap; forced coin-flip tiebreak beyond this
_P0_MIN, _P0_MAX = 0.55, 0.85
_TARGET_MEAN_MIN, _TARGET_MEAN_MAX = 0.2, 1.2  # runs per half-inning
_STARTER_MULT_MIN, _STARTER_MULT_MAX = 0.6, 1.6
_BULLPEN_MULT_MIN, _BULLPEN_MULT_MAX = 0.7, 1.5
_Q_BOUNDS = (1e-6, 0.98)


@dataclass(frozen=True)
class BaseballParams(SportParams):
    """Baseball simulation parameters for one team.

    Per-game run rates feed the odds-ratio expected-runs blend; the pitching
    rates feed the bullpen-phase multiplier (``team_era`` is the fallback when
    ``bullpen_era`` is unavailable). ``team_fip`` is carried for hashing and
    debugging — the starter multiplier uses the announced starter's FIP from
    GameContext, not the team aggregate.
    """

    runs_scored_per_game: float
    runs_allowed_per_game: float
    team_era: float
    team_fip: float
    bullpen_era: float


def map_baseball_stats(stats: TeamStats) -> BaseballParams:
    """Convert a statistics-service team stats response into baseball parameters.

    Empty baseball blocks fall back to MLB league averages, mirroring the NBA
    fallbacks in the basketball mapper (NCAA_BSB is dormant; populated NCAA
    stat blocks flow through the simulator's configured league scoring rate).
    """
    baseball = stats.baseball
    team_era = baseball.team_era if baseball.team_era > 0 else lg.MLB_LEAGUE_ERA
    return BaseballParams(
        runs_scored_per_game=(
            baseball.runs_scored_per_game if baseball.runs_scored_per_game > 0 else lg.MLB_RUNS_PER_GAME
        ),
        runs_allowed_per_game=(
            baseball.runs_allowed_per_game if baseball.runs_allowed_per_game > 0 else lg.MLB_RUNS_PER_GAME
        ),
        team_era=team_era,
        team_fip=baseball.team_fip if baseball.team_fip > 0 else lg.MLB_LEAGUE_FIP,
        bullpen_era=baseball.bullpen_era if baseball.bullpen_era > 0 else team_era,
    )


def _zero_modified_geometric_pmf(p0: float, q: float) -> npt.NDArray[np.float64]:
    """PMF over 0..10: P(0) = p0; the k >= 1 tail is Geom(1-q) truncated at 10.

    The tail is renormalized to mass ``1 - p0`` so P(0) stays exactly p0
    regardless of the truncation.
    """
    pmf = np.empty(_SUPPORT, dtype=np.float64)
    tail = (1.0 - q) * q ** np.arange(_MAX_RUNS_PER_HALF_INNING, dtype=np.float64)
    tail /= tail.sum()
    pmf[0] = p0
    pmf[1:] = (1.0 - p0) * tail
    return pmf


def _half_inning_pmf(target_mean: float, league_mean: float) -> npt.NDArray[np.float64]:
    """Zero-modified geometric half-inning PMF calibrated to ``target_mean``.

    The target is clamped to [0.2, 1.2] (the range over which the p0 band
    keeps calibration feasible and over-dispersed); p0 follows the target so
    hot offenses score via fewer zero innings as well as bigger crooked
    numbers, and the bisection then solves for q. 80 iterations at a 1e-9
    tolerance leave the achieved mean well within 1e-6 of the target.
    """
    target = float(np.clip(target_mean, _TARGET_MEAN_MIN, _TARGET_MEAN_MAX))
    p0 = float(np.clip(lg.BASEBALL_P0_BASE - lg.BASEBALL_P0_ALPHA * (target - league_mean), _P0_MIN, _P0_MAX))
    return calibrate_distribution(
        lambda q: _zero_modified_geometric_pmf(p0, q),
        target,
        _Q_BOUNDS,
        max_iterations=80,
        tolerance=1e-9,
    )


@dataclass(frozen=True)
class _BattingModel:
    """Precomputed half-inning distributions for one batting team."""

    starter_pmf: npt.NDArray[np.float64]
    starter_cdf: npt.NDArray[np.float64]
    bullpen_pmf: npt.NDArray[np.float64]
    bullpen_cdf: npt.NDArray[np.float64]

    def cdf_for_inning(self, inning: int) -> npt.NDArray[np.float64]:
        return self.starter_cdf if inning <= _STARTER_INNINGS else self.bullpen_cdf


def _config_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value) if isinstance(value, int | float) else default


class BaseballSimulator(GameSimulator):
    """Vectorized half-inning runs simulator shared by MLB and NCAA_BSB."""

    def __init__(self, plugin_config: dict[str, object] | None = None) -> None:
        config = plugin_config or {}
        self._league_runs_per_game = _config_float(config, "league_runs_per_game", lg.MLB_RUNS_PER_GAME)
        self._league_fip = _config_float(config, "league_fip", lg.MLB_LEAGUE_FIP)
        self._league_era = _config_float(config, "league_era", lg.MLB_LEAGUE_ERA)
        league = config.get("league")
        self._league_override = league if isinstance(league, str) else None
        self._league = self._league_override or "MLB"
        self._home_batting: _BattingModel | None = None
        self._away_batting: _BattingModel | None = None
        # Diagnostics from the most recent simulate_games call.
        self._last_extra_inning_games = 0
        self._last_forced_tiebreaks = 0

    def _starter_multiplier(self, starter_fip: float | None) -> float:
        if starter_fip is None:
            return 1.0
        return float(np.clip(starter_fip / self._league_fip, _STARTER_MULT_MIN, _STARTER_MULT_MAX))

    def _bullpen_multiplier(self, opponent: BaseballParams) -> float:
        era = opponent.bullpen_era if opponent.bullpen_era > 0 else opponent.team_era
        if era <= 0:
            return 1.0
        return float(np.clip(era / self._league_era, _BULLPEN_MULT_MIN, _BULLPEN_MULT_MAX))

    def _batting_model(
        self, batting: BaseballParams, opponent: BaseballParams, opp_starter_fip: float | None
    ) -> _BattingModel:
        base = (batting.runs_scored_per_game * opponent.runs_allowed_per_game / self._league_runs_per_game) / 9.0
        league_mean = self._league_runs_per_game / 9.0
        starter_pmf = _half_inning_pmf(base * self._starter_multiplier(opp_starter_fip), league_mean)
        bullpen_pmf = _half_inning_pmf(base * self._bullpen_multiplier(opponent), league_mean)
        return _BattingModel(
            starter_pmf=starter_pmf,
            starter_cdf=np.cumsum(starter_pmf),
            bullpen_pmf=bullpen_pmf,
            bullpen_cdf=np.cumsum(bullpen_pmf),
        )

    def set_parameters(self, home_params: SportParams, away_params: SportParams, context: GameContext) -> None:
        if not isinstance(home_params, BaseballParams) or not isinstance(away_params, BaseballParams):
            raise TypeError("BaseballSimulator requires BaseballParams for both teams")
        if self._league_override is None:
            self._league = context.league
        # Each batting team faces the OPPOSING pitching staff: home batters
        # see the away starter/bullpen and vice versa.
        self._home_batting = self._batting_model(home_params, away_params, context.away_starter_fip)
        self._away_batting = self._batting_model(away_params, home_params, context.home_starter_fip)

    def _models(self) -> tuple[_BattingModel, _BattingModel]:
        if self._home_batting is None or self._away_batting is None:
            raise RuntimeError("set_parameters must be called before simulating")
        return self._home_batting, self._away_batting

    @staticmethod
    def _draw_half_inning(rng: np.random.Generator, cdf: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.int64]:
        return np.searchsorted(cdf, rng.random(n), side="right").astype(np.int64)

    def _simulate_regulation(
        self, rng: np.random.Generator, n: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Draw nine innings as (n, 9) per-inning run matrices, bottom-9 skip applied.

        Draws follow game order (top then bottom each inning, 18 draws
        total). The bottom of the 9th is drawn for every game to keep the RNG
        stream deterministic, then zeroed where the home team already leads
        after 8 1/2 — home walk-off innings are otherwise counted in full
        (documented approximation).
        """
        home, away = self._models()
        home_by_inning = np.zeros((n, _REGULATION_INNINGS), dtype=np.int64)
        away_by_inning = np.zeros((n, _REGULATION_INNINGS), dtype=np.int64)
        for inning in range(1, _REGULATION_INNINGS + 1):
            away_by_inning[:, inning - 1] = self._draw_half_inning(rng, away.cdf_for_inning(inning), n)
            home_by_inning[:, inning - 1] = self._draw_half_inning(rng, home.cdf_for_inning(inning), n)
        home_leads_after_eight_and_a_half = home_by_inning[:, :-1].sum(axis=1) > away_by_inning.sum(axis=1)
        home_by_inning[home_leads_after_eight_and_a_half, -1] = 0
        return home_by_inning, away_by_inning

    def simulate_games(self, rng: np.random.Generator, n: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        home, away = self._models()
        home_by_inning, away_by_inning = self._simulate_regulation(rng, n)
        home_scores = home_by_inning.sum(axis=1)
        away_scores = away_by_inning.sum(axis=1)

        # Extra innings: full innings at bullpen-phase rates for the tied
        # subset until every game is decided (no draws, no walk-off
        # truncation, no ghost runner).
        tied = home_scores == away_scores
        self._last_extra_inning_games = int(tied.sum())
        self._last_forced_tiebreaks = 0
        inning = _REGULATION_INNINGS
        while bool(tied.any()):
            inning += 1
            indices = np.flatnonzero(tied)
            if inning > _MAX_INNINGS:
                # Safety valve: award a coin-flip run. Never reached at
                # realistic parameters (asserted in tests).
                home_wins = rng.random(indices.size) < 0.5
                home_scores[indices[home_wins]] += 1
                away_scores[indices[~home_wins]] += 1
                self._last_forced_tiebreaks = int(indices.size)
                break
            away_scores[indices] += self._draw_half_inning(rng, away.bullpen_cdf, indices.size)
            home_scores[indices] += self._draw_half_inning(rng, home.bullpen_cdf, indices.size)
            tied = home_scores == away_scores

        return home_scores.astype(np.int32), away_scores.astype(np.int32)

    def simulate_game(self, rng: np.random.Generator) -> GameResult:
        home, away = self.simulate_games(rng, 1)
        return GameResult(home_score=int(home[0]), away_score=int(away[0]), metadata={})

    def get_sport(self) -> str:
        return "BASEBALL"

    def get_league(self) -> str:
        return self._league
