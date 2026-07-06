"""Drive-based football simulator (Phase 6 Wave 3).

Implements the football plugin from components/simulation-engine.md per
ADR-018: drive-level granularity is the coarsest unit that still yields
correct score, margin, and total shapes for the Phase 6 bet types
(play-level simulation is deferred to Phase 7 with player props).

Model:

- **Drives per game.** Both teams get the same number of offensive drives,
  drawn per game as ``round(Normal(mu, sigma))`` clipped to league bounds
  (NFL mu 10.9 sigma 1.2 clip [7, 16]; college mu 12.5 sigma 1.6 clip
  [8, 18]; league_averages.py). ``mu`` is the average of the two teams'
  drives-per-game rates.
- **Drive outcome.** Each drive resolves categorically over {0, 3, 7}
  points. The expected points per drive is the odds-ratio blend
  ``ppd_off_team x ppd_def_opp / league_ppd`` (league_ppd: NFL 1.95, college
  2.15), clamped to [1.0, 3.2]. The TD:FG probability ratio follows the
  target like baseball's p0 rule:
  ``r = clip(R_BASE + R_ALPHA * (target - league_ppd), 0.05, 4.0)``
  (``R_BASE = 1.35`` so ``p_td / p_fg ~= 1.35`` at league average — hot
  offenses score more of their points as touchdowns), and the shared
  bisection calibrator then solves for ``s = P(score on a drive)`` with
  ``pmf = [1-s, ..., s/(1+r) at 3, ..., s*r/(1+r) at 7]`` so the PMF mean
  matches the target to well under 1e-6. Across the clamped target range
  ``s`` stays inside [0.25, 0.55] for both league baselines (asserted in
  tests). Key numbers 3 and 7 emerge naturally from the score quantization.
- **Home advantage.** A margin shift of ``hfa_margin_points`` (NFL 2.2,
  college 3.0; 0 at neutral sites) is split as +/-(hfa/2) total points per
  side, spread across drives via the per-drive target
  (``+/- hfa / 2 / mu_drives``).
- **Overtime.** NFL (``ot_ties_allowed``): tied regulation games play ONE
  both-possess exchange at an elevated per-drive target (x1.4), then — if
  still tied — one sudden-death drive pair; a tie after that STANDS (a tied
  final grades a two-way moneyline as PUSH). The tie rate lands at ~0.3-1%
  for even matchups. NCAA: alternating single-drive rounds at the elevated
  rate repeat until decided — no college ties.

Documented approximations: safeties and two-point conversions are folded
into the {0, 3, 7} quantization, turnovers appear only implicitly as
scoreless drives, the sudden-death pair compares full drive outcomes rather
than stopping at the first score, and the game clock / end-of-half drives
are not modeled beyond the drive-count distribution.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from simulation_engine.clients.statistics import TeamStats
from simulation_engine.core import league_averages as lg
from simulation_engine.core.calibrate import calibrate_distribution
from simulation_engine.core.framework import GameResult, GameSimulator
from simulation_engine.core.params import GameContext, SportParams

_MAX_POINTS_PER_DRIVE = 7
_SUPPORT = _MAX_POINTS_PER_DRIVE + 1
_FG_POINTS = 3
_TD_POINTS = 7
_TARGET_PPD_MIN, _TARGET_PPD_MAX = 1.0, 3.2  # calibratable band; keeps P(score) in [0.25, 0.55]
_SCORE_PROB_BOUNDS = (0.01, 0.7)  # bisection bounds for s = P(score on a drive)
_MAX_OT_ROUNDS = 60  # NCAA safety cap; forced field-goal tiebreak beyond this


@dataclass(frozen=True)
class FootballParams(SportParams):
    """Football simulation parameters for one team.

    The points-per-drive rates feed the odds-ratio blend; drives_per_game
    feeds the pace draw. Points per game and EPA rates are carried for
    hashing and debugging (EPA is absent for NCAA_FB, which uses SP+
    upstream; 0.0 means league average).
    """

    points_per_game: float
    points_allowed_per_game: float
    drives_per_game: float
    points_per_drive_off: float
    points_per_drive_def: float
    epa_per_play_off: float
    epa_per_play_def: float


def map_football_stats(stats: TeamStats) -> FootballParams:
    """Convert a statistics-service team stats response into football parameters.

    Empty football blocks fall back to NFL league averages, mirroring the
    other sport mappers (NCAA_FB stat blocks flow through the simulator's
    configured college league constants).
    """
    football = stats.football
    drives = football.drives_per_game if football.drives_per_game > 0 else lg.NFL_DRIVES_PER_TEAM_MU
    league_ppg = lg.NFL_POINTS_PER_DRIVE * lg.NFL_DRIVES_PER_TEAM_MU
    return FootballParams(
        points_per_game=football.points_per_game if football.points_per_game > 0 else league_ppg,
        points_allowed_per_game=(
            football.points_allowed_per_game if football.points_allowed_per_game > 0 else league_ppg
        ),
        drives_per_game=drives,
        points_per_drive_off=(
            football.points_per_drive_off if football.points_per_drive_off > 0 else lg.NFL_POINTS_PER_DRIVE
        ),
        points_per_drive_def=(
            football.points_per_drive_def if football.points_per_drive_def > 0 else lg.NFL_POINTS_PER_DRIVE
        ),
        epa_per_play_off=football.epa_per_play_off,
        epa_per_play_def=football.epa_per_play_def,
    )


def _td_fg_ratio(target: float, league_ppd: float) -> float:
    """TD:FG probability ratio tied to the points-per-drive target.

    ~1.35 at a league-average target and rising with it (better offenses
    convert drives into touchdowns rather than settling for field goals);
    the slope keeps P(score) inside [0.25, 0.55] across the clamped range.
    """
    ratio = lg.FOOTBALL_TD_FG_RATIO_BASE + lg.FOOTBALL_TD_FG_RATIO_ALPHA * (target - league_ppd)
    return float(np.clip(ratio, lg.FOOTBALL_TD_FG_RATIO_MIN, lg.FOOTBALL_TD_FG_RATIO_MAX))


def _drive_outcome_pmf(score_prob: float, ratio: float) -> npt.NDArray[np.float64]:
    """PMF over 0..7 points with mass only at {0, 3, 7}: P(score) = s, p_td/p_fg = ratio."""
    pmf = np.zeros(_SUPPORT, dtype=np.float64)
    pmf[0] = 1.0 - score_prob
    pmf[_FG_POINTS] = score_prob / (1.0 + ratio)
    pmf[_TD_POINTS] = score_prob * ratio / (1.0 + ratio)
    return pmf


def _drive_pmf(target_ppd: float, league_ppd: float) -> npt.NDArray[np.float64]:
    """Drive-outcome PMF bisection-calibrated so its mean hits ``target_ppd``.

    The ratio rule fixes p_td/p_fg from the (clamped) target, leaving
    P(score) as the single free parameter; the PMF mean is linear in it, so
    80 bisection iterations at 1e-9 tolerance land well within 1e-6.
    """
    target = float(np.clip(target_ppd, _TARGET_PPD_MIN, _TARGET_PPD_MAX))
    ratio = _td_fg_ratio(target, league_ppd)
    return calibrate_distribution(
        lambda s: _drive_outcome_pmf(s, ratio),
        target,
        _SCORE_PROB_BOUNDS,
        max_iterations=80,
        tolerance=1e-9,
    )


@dataclass(frozen=True)
class _DriveModel:
    """Precomputed drive-outcome distributions for one offense."""

    regulation_pmf: npt.NDArray[np.float64]
    regulation_cdf: npt.NDArray[np.float64]
    overtime_pmf: npt.NDArray[np.float64]
    overtime_cdf: npt.NDArray[np.float64]


def _config_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def _config_bool(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    return value if isinstance(value, bool) else default


class FootballSimulator(GameSimulator):
    """Vectorized drive-based simulator shared by NFL and NCAA_FB."""

    def __init__(self, plugin_config: dict[str, object] | None = None) -> None:
        config = plugin_config or {}
        self._drives_mu_default = _config_float(config, "drives_mu", lg.NFL_DRIVES_PER_TEAM_MU)
        self._drives_sigma = _config_float(config, "drives_sigma", lg.NFL_DRIVES_SIGMA)
        self._drives_clip_min = _config_float(config, "drives_clip_min", lg.NFL_DRIVES_CLIP_MIN)
        self._drives_clip_max = _config_float(config, "drives_clip_max", lg.NFL_DRIVES_CLIP_MAX)
        self._league_ppd = _config_float(config, "league_points_per_drive", lg.NFL_POINTS_PER_DRIVE)
        self._hfa_margin_points = _config_float(config, "hfa_margin_points", lg.NFL_HFA_MARGIN_POINTS)
        self._ot_ties_allowed = _config_bool(config, "ot_ties_allowed", True)
        self._ot_multiplier = _config_float(config, "ot_target_multiplier", lg.FOOTBALL_OT_TARGET_MULTIPLIER)
        league = config.get("league")
        self._league_override = league if isinstance(league, str) else None
        self._league = self._league_override or "NFL"
        self._drives_mu = self._drives_mu_default
        self._home_model: _DriveModel | None = None
        self._away_model: _DriveModel | None = None
        # Diagnostics from the most recent simulate_games call.
        self._last_regulation_ties = 0
        self._last_standing_ties = 0
        self._last_forced_tiebreaks = 0

    def _drive_model(self, target_ppd: float) -> _DriveModel:
        regulation = _drive_pmf(target_ppd, self._league_ppd)
        overtime = _drive_pmf(target_ppd * self._ot_multiplier, self._league_ppd)
        return _DriveModel(
            regulation_pmf=regulation,
            regulation_cdf=np.cumsum(regulation),
            overtime_pmf=overtime,
            overtime_cdf=np.cumsum(overtime),
        )

    def set_parameters(self, home_params: SportParams, away_params: SportParams, context: GameContext) -> None:
        if not isinstance(home_params, FootballParams) or not isinstance(away_params, FootballParams):
            raise TypeError("FootballSimulator requires FootballParams for both teams")
        if self._league_override is None:
            self._league = context.league

        self._drives_mu = (home_params.drives_per_game + away_params.drives_per_game) / 2.0
        # HFA is a margin shift split +/- (hfa / 2) total points per side,
        # spread across the expected number of drives via the target.
        hfa = 0.0 if context.neutral_site else self._hfa_margin_points
        shift_per_drive = (hfa / 2.0) / self._drives_mu
        home_target = home_params.points_per_drive_off * away_params.points_per_drive_def / self._league_ppd
        away_target = away_params.points_per_drive_off * home_params.points_per_drive_def / self._league_ppd
        self._home_model = self._drive_model(home_target + shift_per_drive)
        self._away_model = self._drive_model(away_target - shift_per_drive)

    def _models(self) -> tuple[_DriveModel, _DriveModel]:
        if self._home_model is None or self._away_model is None:
            raise RuntimeError("set_parameters must be called before simulating")
        return self._home_model, self._away_model

    def _draw_drive_counts(self, rng: np.random.Generator, n: int) -> npt.NDArray[np.int64]:
        counts = np.clip(
            np.rint(rng.normal(self._drives_mu, self._drives_sigma, size=n)),
            self._drives_clip_min,
            self._drives_clip_max,
        )
        return counts.astype(np.int64)

    @staticmethod
    def _draw_drives(rng: np.random.Generator, cdf: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.int64]:
        """n single-drive point outcomes (values in {0, 3, 7})."""
        return np.searchsorted(cdf, rng.random(n), side="right").astype(np.int64)

    @staticmethod
    def _sample_game_points(
        rng: np.random.Generator, cdf: npt.NDArray[np.float64], drive_counts: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """Per-game point totals: per-drive categorical draws with per-game drive-count masking."""
        n = len(drive_counts)
        max_drives = int(drive_counts.max())
        points = np.searchsorted(cdf, rng.random((n, max_drives)), side="right").astype(np.int64)
        mask = np.arange(max_drives)[np.newaxis, :] < drive_counts[:, np.newaxis]
        return np.asarray((points * mask).sum(axis=1), dtype=np.int64)

    def _resolve_overtime_nfl(
        self,
        rng: np.random.Generator,
        home_scores: npt.NDArray[np.int64],
        away_scores: npt.NDArray[np.int64],
    ) -> None:
        """One both-possess exchange, then one sudden-death drive pair; remaining ties STAND."""
        home, away = self._models()
        tied_idx = np.flatnonzero(home_scores == away_scores)
        if tied_idx.size:
            home_scores[tied_idx] += self._draw_drives(rng, home.overtime_cdf, tied_idx.size)
            away_scores[tied_idx] += self._draw_drives(rng, away.overtime_cdf, tied_idx.size)
            still_idx = tied_idx[home_scores[tied_idx] == away_scores[tied_idx]]
            if still_idx.size:
                home_scores[still_idx] += self._draw_drives(rng, home.overtime_cdf, still_idx.size)
                away_scores[still_idx] += self._draw_drives(rng, away.overtime_cdf, still_idx.size)
        self._last_standing_ties = int(np.sum(home_scores == away_scores))

    def _resolve_overtime_ncaa(
        self,
        rng: np.random.Generator,
        home_scores: npt.NDArray[np.int64],
        away_scores: npt.NDArray[np.int64],
    ) -> None:
        """Alternating single-drive rounds at the elevated rate until decided — no ties."""
        home, away = self._models()
        tied = home_scores == away_scores
        rounds = 0
        while bool(tied.any()):
            rounds += 1
            indices = np.flatnonzero(tied)
            if rounds > _MAX_OT_ROUNDS:
                # Safety valve: award a field goal to a coin-flip winner. Never
                # reached at realistic parameters (asserted in tests).
                home_wins = rng.random(indices.size) < 0.5
                home_scores[indices[home_wins]] += _FG_POINTS
                away_scores[indices[~home_wins]] += _FG_POINTS
                self._last_forced_tiebreaks = int(indices.size)
                break
            home_scores[indices] += self._draw_drives(rng, home.overtime_cdf, indices.size)
            away_scores[indices] += self._draw_drives(rng, away.overtime_cdf, indices.size)
            tied = home_scores == away_scores
        self._last_standing_ties = 0

    def simulate_games(self, rng: np.random.Generator, n: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        home, away = self._models()
        drive_counts = self._draw_drive_counts(rng, n)
        home_scores = self._sample_game_points(rng, home.regulation_cdf, drive_counts)
        away_scores = self._sample_game_points(rng, away.regulation_cdf, drive_counts)

        self._last_regulation_ties = int(np.sum(home_scores == away_scores))
        self._last_forced_tiebreaks = 0
        if self._ot_ties_allowed:
            self._resolve_overtime_nfl(rng, home_scores, away_scores)
        else:
            self._resolve_overtime_ncaa(rng, home_scores, away_scores)

        return home_scores.astype(np.int32), away_scores.astype(np.int32)

    def simulate_game(self, rng: np.random.Generator) -> GameResult:
        home, away = self.simulate_games(rng, 1)
        return GameResult(home_score=int(home[0]), away_score=int(away[0]), metadata={})

    def get_sport(self) -> str:
        return "FOOTBALL"

    def get_league(self) -> str:
        return self._league
