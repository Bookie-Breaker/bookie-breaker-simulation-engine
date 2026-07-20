"""Map SimulationOutput to the API contract's result and distribution shapes."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from simulation_engine.api.models import (
    Distribution,
    Percentiles,
    PlayerPropsEntry,
    PlayerStatBlock,
    SimulationResultData,
)
from simulation_engine.core.params import PlayerRates
from simulation_engine.core.player_rates import YES_NO_STAT_KEYS
from simulation_engine.core.runner import SimulationOutput

_PERCENTILE_POINTS = (10, 25, 50, 75, 90)


def _percentiles(values: npt.NDArray[np.int32]) -> dict[str, int]:
    return {str(p): int(round(float(np.percentile(values, p)))) for p in _PERCENTILE_POINTS}


def build_result(output: SimulationOutput, result_id: str) -> SimulationResultData:
    return SimulationResultData(
        id=result_id,
        home_win_probability=round(output.home_win_prob, 4),
        away_win_probability=round(output.away_win_prob, 4),
        draw_probability=round(output.draw_prob, 4),
        mean_home_score=round(float(np.mean(output.home_scores)), 2),
        mean_away_score=round(float(np.mean(output.away_scores)), 2),
        mean_total=round(output.total_mean, 2),
        mean_margin=round(output.margin_mean, 2),
        # Keys are home handicaps from the home perspective: "-3.5" means home -3.5
        spread_cover_probabilities={f"{line:+.1f}": round(p, 4) for line, p in sorted(output.spread_covers.items())},
        total_over_probabilities={f"{line:.1f}": round(p, 4) for line, p in sorted(output.total_overs.items())},
        # Integer lines only; half-point lines cannot push and are omitted
        spread_push_probabilities={f"{line:+.1f}": round(p, 4) for line, p in sorted(output.spread_pushes.items())},
        total_push_probabilities={f"{line:.1f}": round(p, 4) for line, p in sorted(output.total_pushes.items())},
        percentiles=Percentiles(margin=_percentiles(output.margins), total=_percentiles(output.totals)),
    )


def _distribution(values: npt.NDArray[np.int32]) -> Distribution:
    unique, counts = np.unique(values, return_counts=True)
    n = len(values)
    return Distribution(
        values={str(int(v)): round(float(c) / n, 4) for v, c in zip(unique, counts, strict=True)},
        mean=round(float(np.mean(values)), 2),
        std_dev=round(float(np.std(values, ddof=1)), 2),
        min=int(values.min()),
        max=int(values.max()),
    )


def build_distributions(output: SimulationOutput) -> dict[str, Distribution]:
    return {
        "home_score": _distribution(output.home_scores),
        "away_score": _distribution(output.away_scores),
        "margin": _distribution(output.margins),
        "total": _distribution(output.totals),
    }


#: Half-step count lines around the mean: 3 steps of 0.5 each side.
_LINE_STEPS = 3
_LINE_STEP = 0.5


def default_line_grid(mean: float) -> list[float]:
    """Half-point lines centered on the mean (Phase 7 Wave 3).

    ``floor(mean) + 0.5`` anchors the grid on the nearest half-point line,
    then +/- 3 whole-point steps around it; lines below 0.5 are dropped
    (P(count > negative line) is trivially 1.0 for count stats).
    """
    center = float(np.floor(mean)) + _LINE_STEP
    lines = [center + k for k in range(-_LINE_STEPS, _LINE_STEPS + 1)]
    return [line for line in lines if line > 0.0]


def _player_stat_block(
    stat_key: str, values: npt.NDArray[np.int32], line_grid_fn: Callable[[float], list[float]]
) -> PlayerStatBlock:
    distribution = _distribution(values)
    if stat_key in YES_NO_STAT_KEYS:
        # YES/NO markets settle on count > 0; no line grid applies.
        return PlayerStatBlock(distribution=distribution, yes_probability=round(float(np.mean(values > 0)), 4))
    over_probabilities = {
        f"{line:.1f}": round(float(np.mean(values > line)), 4) for line in line_grid_fn(float(np.mean(values)))
    }
    return PlayerStatBlock(distribution=distribution, over_probabilities=over_probabilities)


def build_player_distributions(
    output: SimulationOutput,
    roster: dict[str, PlayerRates],
    line_grid_fn: Callable[[float], list[float]] = default_line_grid,
) -> dict[str, PlayerPropsEntry]:
    """Per-player stat distributions and over-probability grids (Phase 7 Wave 3).

    ``roster`` maps player UUID -> PlayerRates (for name/team metadata);
    players present in the output but missing from the roster are skipped
    defensively. OVER_UNDER stats get half-point lines from ``line_grid_fn``
    around each stat's mean; YES_NO stats get ``yes_probability`` instead.
    """
    players: dict[str, PlayerPropsEntry] = {}
    for player_id, stats in output.player_stats.items():
        meta = roster.get(player_id)
        if meta is None:
            continue
        players[player_id] = PlayerPropsEntry(
            name=meta.name,
            team=meta.team,
            stats={key: _player_stat_block(key, values, line_grid_fn) for key, values in stats.items()},
        )
    return players
