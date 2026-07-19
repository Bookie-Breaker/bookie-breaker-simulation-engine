"""Same-game parlay correlation artifacts over the canonical leg vocabulary (Phase 7 Wave 1).

The Monte Carlo runner holds index-aligned per-iteration sample arrays
(``SimulationOutput.margins`` / ``totals``), so the joint structure between the
betting legs of one game exists in memory at run time. This module turns those
raw arrays into a cacheable artifact:

- a pairwise phi/Pearson correlation matrix over a canonical leg vocabulary,
- empirical marginals for each leg, and
- a bit-packed boolean leg matrix (``np.packbits`` over legs x iterations) so
  the empirical joint probability of an ARBITRARY subset of legs stays
  computable at read time after the raw score arrays are gone.

Canonical leg keys (cross-service contract; the agent depends on these exact
formats, lines rendered with ``%g``):

- ``MONEYLINE:HOME`` / ``MONEYLINE:AWAY`` / ``MONEYLINE:DRAW``
- ``SPREAD:HOME:{line}`` / ``SPREAD:AWAY:{line}`` -- a HOME line x covers when
  ``margin + x > 0``; the AWAY side is the negation (away covers at line x when
  ``-margin + x > 0``).
- ``TOTAL:OVER:{line}`` / ``TOTAL:UNDER:{line}`` -- OVER is ``total > line``,
  UNDER is ``total < line``.

Push handling: pushes are EXCLUDED, i.e. a push counts as ``False`` (strict
inequalities everywhere). On integer lines ``margin == -line`` /
``total == line`` iterations satisfy neither side of the market; on half-point
lines pushes cannot occur, so the two sides are exact complements.
"""

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

import numpy as np
import numpy.typing as npt

from simulation_engine.core.runner import SimulationOutput

_MARKETS_TWO_PART = frozenset({"MONEYLINE"})
_SIDES = {
    "MONEYLINE": frozenset({"HOME", "AWAY", "DRAW"}),
    "SPREAD": frozenset({"HOME", "AWAY"}),
    "TOTAL": frozenset({"OVER", "UNDER"}),
}
_LEG_FORMAT_HINT = "expected 'MONEYLINE:HOME|AWAY|DRAW', 'SPREAD:HOME|AWAY:<line>', or 'TOTAL:OVER|UNDER:<line>'"


class UnknownLegError(ValueError):
    """Raised for leg keys outside the canonical vocabulary or the stored artifact."""


def format_line(line: float) -> str:
    """Render a line with ``%g`` (``-1.5``, ``2.5``, ``220.5``), normalizing ``-0.0``."""
    return f"{0.0 if line == 0 else line:g}"


def _parse_leg(leg_key: str) -> tuple[str, str, float | None]:
    """Split a canonical leg key into (market, side, line); line is None for moneylines."""
    parts = leg_key.split(":")
    market = parts[0]
    sides = _SIDES.get(market)
    if sides is None:
        raise UnknownLegError(f"Unknown leg key {leg_key!r}: unknown market {market!r}; {_LEG_FORMAT_HINT}")
    if market in _MARKETS_TWO_PART:
        if len(parts) != 2 or parts[1] not in sides:
            raise UnknownLegError(f"Unknown leg key {leg_key!r}; {_LEG_FORMAT_HINT}")
        return market, parts[1], None
    if len(parts) != 3 or parts[1] not in sides:
        raise UnknownLegError(f"Unknown leg key {leg_key!r}; {_LEG_FORMAT_HINT}")
    try:
        line = float(parts[2])
    except ValueError:
        raise UnknownLegError(f"Unknown leg key {leg_key!r}: line {parts[2]!r} is not a number") from None
    if not math.isfinite(line):
        raise UnknownLegError(f"Unknown leg key {leg_key!r}: line must be finite")
    return market, parts[1], line


def leg_vector(output: SimulationOutput, leg_key: str) -> npt.NDArray[np.bool_]:
    """Per-iteration boolean vector for one canonical leg key.

    Pushes are excluded (count as False): all comparisons are strict, so on an
    integer line the iterations landing exactly on the line satisfy neither
    side. Raises :class:`UnknownLegError` for keys outside the vocabulary.
    """
    market, side, line = _parse_leg(leg_key)
    margins = output.margins
    totals = output.totals
    if market == "MONEYLINE":
        if side == "HOME":
            return np.asarray(margins > 0)
        if side == "AWAY":
            return np.asarray(margins < 0)
        return np.asarray(margins == 0)  # DRAW
    assert line is not None
    if market == "SPREAD":
        # HOME line x covers when margin + x > 0; AWAY is the negation.
        if side == "HOME":
            return np.asarray(margins + line > 0)
        return np.asarray(-margins + line > 0)
    # TOTAL
    if side == "OVER":
        return np.asarray(totals > line)
    return np.asarray(totals < line)


def default_legs(output: SimulationOutput, include_draw: bool = False) -> list[str]:
    """Default leg vocabulary for an output: moneylines + the runner's line grids.

    Spread legs come from the ``spread_covers`` keys (home handicaps, HOME
    side) and total legs from the ``total_overs`` keys (OVER side) — the same
    grids the cover-probability dicts already expose. The AWAY/UNDER sides of
    half-point lines are exact complements and are resolved at read time
    without being stored. ``MONEYLINE:DRAW`` is included only when the sport
    has draws (soccer regulation).
    """
    legs = ["MONEYLINE:HOME", "MONEYLINE:AWAY"]
    if include_draw:
        legs.append("MONEYLINE:DRAW")
    legs.extend(f"SPREAD:HOME:{format_line(h)}" for h in sorted(output.spread_covers))
    legs.extend(f"TOTAL:OVER:{format_line(t)}" for t in sorted(output.total_overs))
    return legs


def empirical_joint(output: SimulationOutput, legs: list[str]) -> float:
    """Empirical joint probability: mean of the AND of the legs' boolean vectors."""
    if not legs:
        raise UnknownLegError("At least one leg is required for a joint probability")
    vectors = [leg_vector(output, leg) for leg in legs]
    return float(np.mean(reduce(np.logical_and, vectors)))


def _correlation_matrix(vectors: npt.NDArray[np.bool_]) -> list[list[float]]:
    """Pairwise phi/Pearson correlations, nan-guarded.

    Zero-variance vectors (a leg that is always or never true) yield 0.0
    correlation against everything; the diagonal is forced to exactly 1.0.
    """
    stacked = vectors.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.atleast_2d(np.corrcoef(stacked))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 1.0)
    return [[round(float(v), 6) for v in row] for row in matrix]


@dataclass
class _LegRef:
    """A requested leg resolved against a stored artifact: row index + negation flag."""

    index: int
    negate: bool


@dataclass
class CorrelationArtifact:
    """Cacheable correlation artifact for one simulation run.

    ``packed_matrix`` is ``np.packbits`` of the legs x iterations boolean
    matrix (row-major, each row padded to a whole byte), which keeps
    arbitrary-subset joint probabilities computable at read time without the
    raw score arrays (~n_legs x iterations/8 bytes before compression).
    """

    legs: list[str]
    marginals: dict[str, float]
    matrix: list[list[float]]
    iterations: int
    packed_matrix: bytes
    joint_goal_grid: list[list[float]] | None = field(default=None)

    def to_payload(self) -> dict[str, object]:
        """JSON-serializable payload (packed matrix base64-encoded)."""
        return {
            "legs": self.legs,
            "marginals": self.marginals,
            "matrix": self.matrix,
            "iterations": self.iterations,
            "packed_matrix_b64": base64.b64encode(self.packed_matrix).decode(),
            "joint_goal_grid": self.joint_goal_grid,
        }

    @staticmethod
    def from_payload(payload: Mapping[str, Any]) -> "CorrelationArtifact":
        grid = payload.get("joint_goal_grid")
        return CorrelationArtifact(
            legs=[str(leg) for leg in payload["legs"]],
            marginals={str(k): float(v) for k, v in payload["marginals"].items()},
            matrix=[[float(v) for v in row] for row in payload["matrix"]],
            iterations=int(payload["iterations"]),
            packed_matrix=base64.b64decode(str(payload["packed_matrix_b64"])),
            joint_goal_grid=[[float(v) for v in row] for row in grid] if isinstance(grid, list) else None,
        )

    def _leg_index(self) -> dict[str, int]:
        return {leg: i for i, leg in enumerate(self.legs)}

    def resolve(self, leg_key: str) -> _LegRef:
        """Resolve a canonical leg key to a stored row, negating exact complements.

        On half-point lines (where pushes cannot occur) the opposite side of a
        stored spread/total leg is its exact boolean complement, so e.g.
        ``SPREAD:AWAY:1.5`` resolves to the negation of ``SPREAD:HOME:-1.5``.
        Integer-line complements are NOT exact (pushes) and are not resolved.
        """
        index = self._leg_index().get(leg_key)
        if index is not None:
            return _LegRef(index, False)
        market, side, line = _parse_leg(leg_key)
        complement: str | None = None
        if line is not None and not line.is_integer():
            if market == "SPREAD":
                other = "AWAY" if side == "HOME" else "HOME"
                complement = f"SPREAD:{other}:{format_line(-line)}"
            else:  # TOTAL
                other = "UNDER" if side == "OVER" else "OVER"
                complement = f"TOTAL:{other}:{format_line(line)}"
        if complement is not None:
            comp_index = self._leg_index().get(complement)
            if comp_index is not None:
                return _LegRef(comp_index, True)
        raise UnknownLegError(
            f"Leg {leg_key!r} is not available in the stored artifact "
            f"(stored legs and half-point complements only; stored: {', '.join(self.legs)})"
        )

    def _unpack_row(self, ref: _LegRef) -> npt.NDArray[np.bool_]:
        row_bytes = (self.iterations + 7) // 8
        rows = np.frombuffer(self.packed_matrix, dtype=np.uint8).reshape(len(self.legs), row_bytes)
        bits = np.unpackbits(rows[ref.index], count=self.iterations).astype(np.bool_)
        return np.asarray(~bits if ref.negate else bits)

    def subset(self, legs: list[str]) -> tuple[dict[str, float], list[list[float]], float]:
        """Marginals, correlation submatrix, and empirical joint for a leg subset.

        The joint probability is computed from the packed boolean matrix (the
        AND of the requested legs' vectors), which a pairwise matrix alone
        could not provide for 3+ legs.
        """
        if not legs:
            raise UnknownLegError("At least one leg is required")
        refs = [self.resolve(leg) for leg in legs]
        stored = np.asarray(self.matrix, dtype=np.float64)
        marginals = {
            leg: round(1.0 - self.marginals[self.legs[ref.index]], 6)
            if ref.negate
            else self.marginals[self.legs[ref.index]]
            for leg, ref in zip(legs, refs, strict=True)
        }
        signs = np.array([-1.0 if ref.negate else 1.0 for ref in refs])
        indices = np.array([ref.index for ref in refs])
        submatrix = stored[np.ix_(indices, indices)] * np.outer(signs, signs)
        np.fill_diagonal(submatrix, 1.0)
        vectors = [self._unpack_row(ref) for ref in refs]
        joint = float(np.mean(reduce(np.logical_and, vectors)))
        return marginals, [[round(float(v), 6) for v in row] for row in submatrix], joint


def build_correlation_artifact(
    output: SimulationOutput,
    legs: list[str] | None = None,
    include_draw: bool = False,
    joint_goal_grid: npt.NDArray[np.float64] | None = None,
) -> CorrelationArtifact:
    """Build the correlation artifact from in-memory simulation output.

    Must be called AT RUN TIME while the raw sample arrays are still in
    memory — the artifact (including the packed boolean matrix) is everything
    the read path retains. ``joint_goal_grid`` is the sport plugin's analytic
    joint score PMF when available (soccer/hockey), passed through verbatim.
    """
    leg_keys = default_legs(output, include_draw=include_draw) if legs is None else legs
    if not leg_keys:
        raise UnknownLegError("At least one leg is required to build a correlation artifact")
    vectors = np.stack([leg_vector(output, leg) for leg in leg_keys])
    marginals = {leg: round(float(np.mean(vec)), 6) for leg, vec in zip(leg_keys, vectors, strict=True)}
    return CorrelationArtifact(
        legs=list(leg_keys),
        marginals=marginals,
        matrix=_correlation_matrix(vectors),
        iterations=output.iterations_run,
        packed_matrix=np.packbits(vectors, axis=1).tobytes(),
        joint_goal_grid=(
            [[round(float(v), 8) for v in row] for row in joint_goal_grid] if joint_goal_grid is not None else None
        ),
    )
