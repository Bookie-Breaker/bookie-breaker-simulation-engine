"""Sport-agnostic chunked Monte Carlo runner.

Runs the simulator in chunks of ``convergence_check_interval`` iterations
through the plugin's vectorized batch hook, checking convergence between
chunks. A single RNG seeded once makes runs reproducible: identical seed and
parameters produce byte-identical score arrays regardless of early stopping,
because chunk boundaries are deterministic.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from simulation_engine.core.convergence import ConvergenceTracker
from simulation_engine.core.framework import GameSimulator
from simulation_engine.core.params import GameContext, SportParams


@dataclass(frozen=True)
class GridConfig:
    """Per-sport radii for the default spread/total line grids.

    The grids enumerate half-point lines within +/- the radius of the mean
    margin (spreads) and mean total (totals). High-scoring sports use wide
    radii (basketball 10/12); low-scoring sports use narrow ones (e.g.
    soccer 3/4).
    """

    spread_radius: int  # half-point handicaps within +/- this of the mean margin
    total_radius: int  # half-point totals within +/- this of the mean total


#: Basketball's grid radii — the pre-Phase-6 hardcoded defaults.
BASKETBALL_GRID_CONFIG = GridConfig(spread_radius=10, total_radius=12)


@dataclass
class SimulationOutput:
    """Aggregated output from N simulation iterations."""

    iterations_run: int
    converged: bool
    convergence_iteration: int | None
    standard_error: float

    home_scores: npt.NDArray[np.int32]
    away_scores: npt.NDArray[np.int32]
    margins: npt.NDArray[np.int32]
    totals: npt.NDArray[np.int32]

    home_win_prob: float
    away_win_prob: float
    draw_prob: float

    margin_mean: float
    margin_std: float
    total_mean: float
    total_std: float

    # {home handicap: P(home covers)} e.g. -3.5 -> P(margin > 3.5)
    spread_covers: dict[float, float] = field(default_factory=dict)
    # {total line: P(over)}
    total_overs: dict[float, float] = field(default_factory=dict)

    # Push probabilities for INTEGER lines only — half-point lines cannot
    # push and are omitted. {home handicap h: P(margin == -h)}.
    spread_pushes: dict[float, float] = field(default_factory=dict)
    # {total line t: P(total == t)} for integer lines only.
    total_pushes: dict[float, float] = field(default_factory=dict)

    elapsed_ms: float = 0.0


def _spread_lines(margin_mean: float, radius: int) -> list[float]:
    center = -round(margin_mean)
    return [center + k + 0.5 for k in range(-radius - 1, radius + 1)]


def _total_lines(total_mean: float, radius: int) -> list[float]:
    center = round(total_mean)
    return [center + k + 0.5 for k in range(-radius - 1, radius + 1)]


def run_monte_carlo(
    simulator: GameSimulator,
    home_params: SportParams,
    away_params: SportParams,
    context: GameContext,
    iterations: int = 10_000,
    convergence_threshold: float = 0.005,
    convergence_check_interval: int = 1_000,
    seed: int | None = None,
    common_spreads: list[float] | None = None,
    common_totals: list[float] | None = None,
    grid_config: GridConfig = BASKETBALL_GRID_CONFIG,
) -> SimulationOutput:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    simulator.set_parameters(home_params, away_params, context)

    home_scores = np.zeros(iterations, dtype=np.int32)
    away_scores = np.zeros(iterations, dtype=np.int32)
    tracker = ConvergenceTracker(se_threshold=convergence_threshold)
    converged = False
    convergence_iteration: int | None = None

    n = 0
    while n < iterations:
        chunk = min(convergence_check_interval, iterations - n)
        chunk_home, chunk_away = simulator.simulate_games(rng, chunk)
        home_scores[n : n + chunk] = chunk_home
        away_scores[n : n + chunk] = chunk_away
        n += chunk

        state = tracker.check(home_scores[:n] - away_scores[:n], home_scores[:n] + away_scores[:n])
        if state.converged and n < iterations:
            converged = True
            convergence_iteration = n
            break
        if state.converged:
            converged = True
            convergence_iteration = n

    home_scores = home_scores[:n]
    away_scores = away_scores[:n]
    margins = home_scores - away_scores
    totals = home_scores + away_scores

    margin_mean = float(np.mean(margins))
    total_mean = float(np.mean(totals))

    spread_line_values = (
        common_spreads if common_spreads is not None else _spread_lines(margin_mean, grid_config.spread_radius)
    )
    total_line_values = (
        common_totals if common_totals is not None else _total_lines(total_mean, grid_config.total_radius)
    )
    spread_covers = {
        # home handicap h covers when margin > -h (home -3.5 needs margin > 3.5)
        float(h): float(np.mean(margins > -h))
        for h in spread_line_values
    }
    total_overs = {float(t): float(np.mean(totals > t)) for t in total_line_values}
    # Pushes exist only on integer lines; half-point lines are omitted entirely.
    spread_pushes = {float(h): float(np.mean(margins == -h)) for h in spread_line_values if float(h).is_integer()}
    total_pushes = {float(t): float(np.mean(totals == t)) for t in total_line_values if float(t).is_integer()}

    return SimulationOutput(
        iterations_run=n,
        converged=converged,
        convergence_iteration=convergence_iteration,
        standard_error=tracker.last_standard_error,
        home_scores=home_scores,
        away_scores=away_scores,
        margins=margins,
        totals=totals,
        home_win_prob=float(np.mean(margins > 0)),
        away_win_prob=float(np.mean(margins < 0)),
        draw_prob=float(np.mean(margins == 0)),
        margin_mean=margin_mean,
        margin_std=float(np.std(margins, ddof=1)),
        total_mean=total_mean,
        total_std=float(np.std(totals, ddof=1)),
        spread_covers=spread_covers,
        total_overs=total_overs,
        spread_pushes=spread_pushes,
        total_pushes=total_pushes,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
