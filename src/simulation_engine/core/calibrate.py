"""Shared bisection calibrator for plugin score distributions (Phase 6 Wave 2).

Several plugins build a discrete PMF from a single free parameter and need the
PMF's mean to hit a target: basketball scales shot-make probabilities so
expected points per possession match blended team ratings, and baseball scales
the geometric tail of its half-inning run distribution so expected runs match
the matchup rate. The bisection loop is identical in both cases, so it lives
here; the defaults reproduce the original basketball loop exactly (40
iterations, 1e-5 tolerance), which keeps pre-Wave-2 basketball output
byte-identical.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def calibrate_distribution(
    build_pmf: Callable[[float], npt.NDArray[np.float64]],
    target_mean: float,
    param_bounds: tuple[float, float],
    *,
    max_iterations: int = 40,
    tolerance: float = 1e-5,
) -> npt.NDArray[np.float64]:
    """Bisect ``build_pmf``'s parameter until the PMF mean hits ``target_mean``.

    ``build_pmf`` maps a scalar parameter to a PMF over the support
    ``0..len(pmf)-1`` whose mean is monotonically increasing in the parameter.
    Returns the last PMF built; if ``target_mean`` lies outside the range
    achievable within ``param_bounds``, the bisection converges to the nearest
    boundary distribution instead of raising.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    low, high = param_bounds
    pmf = build_pmf((low + high) / 2.0)
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        pmf = build_pmf(mid)
        mean = float(np.dot(pmf, np.arange(len(pmf))))
        if abs(mean - target_mean) < tolerance:
            break
        if mean < target_mean:
            low = mid
        else:
            high = mid
    return pmf
