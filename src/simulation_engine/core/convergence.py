"""Convergence diagnostics per algorithms/simulation-algorithms.md.

Two complementary criteria:

1. Standard error of the mean margin (primary, always reported):
   SE = std(margins, ddof=1) / sqrt(N) < threshold. For basketball margins
   (std ~12) this is unreachable at 10k iterations, so it rarely triggers.
2. Probability stability (practical): the key derived probabilities move
   less than 0.002 between consecutive checks, twice in a row.

``converged`` is true when either criterion fires. Checks begin once at
least ``min_iterations`` samples exist.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ConvergenceState:
    converged: bool
    standard_error: float


class ConvergenceTracker:
    def __init__(
        self,
        se_threshold: float,
        stability_tolerance: float = 0.002,
        min_iterations: int = 2_000,
    ) -> None:
        self._se_threshold = se_threshold
        self._stability_tolerance = stability_tolerance
        self._min_iterations = min_iterations
        self._reference_margin_line: float | None = None
        self._reference_total_line: float | None = None
        self._previous_probs: tuple[float, float, float] | None = None
        self._quiet_checks = 0
        self.last_standard_error = float("inf")

    def check(self, margins: npt.NDArray[np.int32], totals: npt.NDArray[np.int32]) -> ConvergenceState:
        n = len(margins)
        se = float(np.std(margins, ddof=1) / np.sqrt(n))
        self.last_standard_error = se

        if n < self._min_iterations:
            return ConvergenceState(converged=False, standard_error=se)

        if se < self._se_threshold:
            return ConvergenceState(converged=True, standard_error=se)

        # Fix reference lines at the first eligible check so consecutive
        # probability comparisons measure the same quantity.
        if self._reference_margin_line is None or self._reference_total_line is None:
            self._reference_margin_line = float(np.median(margins)) + 0.5
            self._reference_total_line = float(np.median(totals)) + 0.5

        probs = (
            float(np.mean(margins > 0)),
            float(np.mean(margins > self._reference_margin_line)),
            float(np.mean(totals > self._reference_total_line)),
        )
        if self._previous_probs is not None:
            max_change = max(abs(a - b) for a, b in zip(probs, self._previous_probs, strict=True))
            if max_change < self._stability_tolerance:
                self._quiet_checks += 1
            else:
                self._quiet_checks = 0
        self._previous_probs = probs

        return ConvergenceState(converged=self._quiet_checks >= 2, standard_error=se)
