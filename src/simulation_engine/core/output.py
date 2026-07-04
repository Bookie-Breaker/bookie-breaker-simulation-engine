"""Map SimulationOutput to the API contract's result and distribution shapes."""

import numpy as np
import numpy.typing as npt

from simulation_engine.api.models import Distribution, Percentiles, SimulationResultData
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
