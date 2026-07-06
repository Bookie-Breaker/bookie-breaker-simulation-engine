"""Vectorized possession-based NBA simulator.

Implements the outcome tree from algorithms/simulation-algorithms.md section
3 (turnover -> shooting foul/FTs -> 3pt/2pt attempt -> and-1 -> miss with
offensive-rebound restart) in closed form: because rebound restarts are
i.i.d. repeats of the same attempt cycle, the per-possession points
distribution is the terminal-outcome cycle PMF divided by (1 - P(restart)).
Games are then sampled as vectorized categorical draws over that PMF, which
keeps 10,000 iterations well under the 10-second DoD budget (~tens of ms).

Deliberate deviations from the doc's pseudocode, documented for review:

- Shot make probabilities blend offense made % with defense allowed % as
  (off + allowed) / 2. The doc's final "fallback" expression
  ``(off * 0.5) + ((1 - def_allowed) * 0.5)`` yields ~0.50 make rates for
  league-average three-point shooting, which is not a plausible 3P%; the
  intent (blend offense and defense) is preserved with sane magnitudes.
- Make probabilities are then scaled so each team's expected points per 100
  possessions matches the blend of its offensive rating and the opponent's
  defensive rating. This anchors simulated means to observed ratings while
  the possession tree supplies the variance shape.
- Foul-accumulation/bonus state and the game clock are not modeled (the doc
  lists both among its simplifying approximations); pace variance is drawn
  per game instead.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from simulation_engine.core import league_averages as lg
from simulation_engine.core.framework import GameResult, GameSimulator
from simulation_engine.core.params import GameContext, SportParams, TeamParams

_MAX_POINTS_PER_POSSESSION = 3
_MAKE_PROB_FLOOR = 0.05
_MAKE_PROB_CEIL = 0.95


@dataclass(frozen=True)
class _PossessionModel:
    """Per-possession points PMF for one offense/defense pairing."""

    pmf: npt.NDArray[np.float64]  # P(points == k) for k in 0..3
    cdf: npt.NDArray[np.float64]

    @property
    def expected_points(self) -> float:
        return float(np.dot(self.pmf, np.arange(len(self.pmf))))


def _binomial_pmf(n: int, p: float) -> list[float]:
    if n == 2:
        return [(1 - p) ** 2, 2 * p * (1 - p), p**2]
    return [(1 - p) ** 3, 3 * p * (1 - p) ** 2, 3 * p**2 * (1 - p), p**3]


def _build_possession_pmf(offense: TeamParams, defense: TeamParams, make_scale: float = 1.0) -> npt.NDArray[np.float64]:
    to_rate = (offense.tov_pct + defense.forced_tov_pct) / 2.0 / 100.0
    p_foul = ((offense.ft_rate + defense.opp_ft_rate) / 2.0) * 0.15
    r3 = offense.three_attempt_rate
    oreb = (offense.oreb_pct + defense.opp_oreb_pct) / 2.0 / 100.0

    def scaled(p: float) -> float:
        return float(np.clip(p * make_scale, _MAKE_PROB_FLOOR, _MAKE_PROB_CEIL))

    p3 = scaled((offense.three_pct + defense.opp_three_pct) / 2.0)
    p2 = scaled((offense.two_pct + defense.opp_two_pct) / 2.0)
    ft = scaled(offense.ft_pct)
    and1 = lg.NBA_AND_ONE_RATE * ft

    shot = (1.0 - to_rate) * (1.0 - p_foul)
    foul = (1.0 - to_rate) * p_foul
    ft2 = _binomial_pmf(2, ft)
    ft3 = _binomial_pmf(3, ft)

    miss_rate = r3 * (1.0 - p3) + (1.0 - r3) * (1.0 - p2)
    p_repeat = shot * miss_rate * oreb

    cycle = np.zeros(_MAX_POINTS_PER_POSSESSION + 1)
    cycle[0] = to_rate + foul * (r3 * ft3[0] + (1 - r3) * ft2[0]) + shot * miss_rate * (1.0 - oreb)
    cycle[1] = foul * (r3 * ft3[1] + (1 - r3) * ft2[1])
    cycle[2] = foul * (r3 * ft3[2] + (1 - r3) * ft2[2]) + shot * (1 - r3) * p2 * (1.0 - and1)
    cycle[3] = foul * r3 * ft3[3] + shot * r3 * p3 + shot * (1 - r3) * p2 * and1

    pmf: npt.NDArray[np.float64] = cycle / (1.0 - p_repeat)
    return pmf


def _calibrated_model(offense: TeamParams, defense: TeamParams, target_ppp: float) -> _PossessionModel:
    """Bisect the make-probability scale so expected points/possession hits the target."""
    low, high = 0.5, 1.6
    pmf = _build_possession_pmf(offense, defense)
    for _ in range(40):
        mid = (low + high) / 2
        pmf = _build_possession_pmf(offense, defense, make_scale=mid)
        expected = float(np.dot(pmf, np.arange(len(pmf))))
        if abs(expected - target_ppp) < 1e-5:
            break
        if expected < target_ppp:
            low = mid
        else:
            high = mid
    return _PossessionModel(pmf=pmf, cdf=np.cumsum(pmf))


class BasketballSimulator(GameSimulator):
    """Possession-based NBA simulator with a vectorized batch path."""

    def __init__(self, plugin_config: dict[str, object] | None = None) -> None:
        config = plugin_config or {}
        home_advantage = config.get("home_advantage", lg.NBA_HOME_ADVANTAGE)
        self._home_advantage = float(home_advantage) if isinstance(home_advantage, int | float) else 1.5
        self._home_model: _PossessionModel | None = None
        self._away_model: _PossessionModel | None = None
        self._game_pace: float = lg.NBA_LEAGUE_AVG_PACE
        self._ot_possessions: int = round(lg.NBA_LEAGUE_AVG_PACE * lg.NBA_OT_POSSESSION_FRACTION)

    def set_parameters(self, home_params: SportParams, away_params: SportParams, context: GameContext) -> None:
        if not isinstance(home_params, TeamParams) or not isinstance(away_params, TeamParams):
            raise TypeError("BasketballSimulator requires TeamParams for both teams")
        slow, fast = sorted((home_params.pace, away_params.pace))
        self._game_pace = (slow * 0.55 + fast * 0.45) * 0.7 + lg.NBA_LEAGUE_AVG_PACE * 0.3
        self._ot_possessions = max(1, round(self._game_pace * lg.NBA_OT_POSSESSION_FRACTION))

        hca = 0.0 if context.neutral_site else self._home_advantage
        home_target = ((home_params.off_rating + away_params.def_rating) / 2.0 + hca / 2.0) / 100.0
        away_target = ((away_params.off_rating + home_params.def_rating) / 2.0 - hca / 2.0) / 100.0
        self._home_model = _calibrated_model(home_params, away_params, home_target)
        self._away_model = _calibrated_model(away_params, home_params, away_target)

    def _models(self) -> tuple[_PossessionModel, _PossessionModel]:
        if self._home_model is None or self._away_model is None:
            raise RuntimeError("set_parameters must be called before simulating")
        return self._home_model, self._away_model

    def _sample_scores(
        self,
        rng: np.random.Generator,
        cdf: npt.NDArray[np.float64],
        possessions: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int32]:
        n = len(possessions)
        max_poss = int(possessions.max())
        draws = rng.random((n, max_poss))
        points = np.searchsorted(cdf, draws, side="right").astype(np.int32)
        mask = np.arange(max_poss)[np.newaxis, :] < possessions[:, np.newaxis]
        return np.asarray((points * mask).sum(axis=1), dtype=np.int32)

    def simulate_games(self, rng: np.random.Generator, n: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        home_model, away_model = self._models()

        possessions = np.clip(np.rint(rng.normal(self._game_pace, lg.NBA_POSSESSION_STD, size=n)), 70, 135).astype(
            np.int64
        )
        home_scores = self._sample_scores(rng, home_model.cdf, possessions)
        away_scores = self._sample_scores(rng, away_model.cdf, possessions)

        # Overtime: resimulate ~5-minute mini-games for tied rows until decided.
        tied = home_scores == away_scores
        while bool(tied.any()):
            n_tied = int(tied.sum())
            ot_poss = np.full(n_tied, self._ot_possessions, dtype=np.int64)
            home_scores[tied] += self._sample_scores(rng, home_model.cdf, ot_poss)
            away_scores[tied] += self._sample_scores(rng, away_model.cdf, ot_poss)
            tied = home_scores == away_scores

        return home_scores, away_scores

    def simulate_game(self, rng: np.random.Generator) -> GameResult:
        home, away = self.simulate_games(rng, 1)
        return GameResult(home_score=int(home[0]), away_score=int(away[0]), metadata={})

    def get_sport(self) -> str:
        return "BASKETBALL"

    def get_league(self) -> str:
        return "NBA"
