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

Player-prop legs (Phase 7 Wave 4, present only when the run captured player
props via ``include_player_props``):

- ``PLAYER_PROP:{player_uuid}:{stat_key}:OVER:{line}`` /
  ``...:UNDER:{line}`` -- OVER_UNDER stats; OVER is ``stat > line``.
- ``PLAYER_PROP:{player_uuid}:{stat_key}:YES`` / ``...:NO`` -- YES_NO stats
  (``player_goal_scorer_anytime``, ``player_anytime_td``); YES is
  ``stat > 0``.

Push handling: pushes are EXCLUDED, i.e. a push counts as ``False`` (strict
inequalities everywhere). On integer lines ``margin == -line`` /
``total == line`` iterations satisfy neither side of the market; on half-point
lines pushes cannot occur, so the two sides are exact complements. Storage is
one-sided (HOME/OVER/YES plus moneylines); the opposite sides of half-point
legs -- and NO for YES legs -- resolve at read time as negated packed rows.
"""

import base64
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

import numpy as np
import numpy.typing as npt

from simulation_engine.core.output import default_line_grid
from simulation_engine.core.player_rates import YES_NO_STAT_KEYS
from simulation_engine.core.runner import SimulationOutput

logger = logging.getLogger(__name__)

_MARKETS_TWO_PART = frozenset({"MONEYLINE"})
_SIDES = {
    "MONEYLINE": frozenset({"HOME", "AWAY", "DRAW"}),
    "SPREAD": frozenset({"HOME", "AWAY"}),
    "TOTAL": frozenset({"OVER", "UNDER"}),
}
_PLAYER_PROP_MARKET = "PLAYER_PROP"
_PLAYER_YES_NO_SIDES = frozenset({"YES", "NO"})
_PLAYER_OVER_UNDER_SIDES = frozenset({"OVER", "UNDER"})
_LEG_FORMAT_HINT = (
    "expected 'MONEYLINE:HOME|AWAY|DRAW', 'SPREAD:HOME|AWAY:<line>', 'TOTAL:OVER|UNDER:<line>', "
    "'PLAYER_PROP:<player_id>:<stat_key>:OVER|UNDER:<line>', or 'PLAYER_PROP:<player_id>:<stat_key>:YES|NO'"
)

#: Max stored OVER lines per player stat (the lines closest to the stat mean).
PLAYER_LEG_LINE_CAP = 3
#: Max stored player legs per artifact (size control; truncation is logged).
MAX_PLAYER_LEGS = 120


class UnknownLegError(ValueError):
    """Raised for leg keys outside the canonical vocabulary or the stored artifact."""


def format_line(line: float) -> str:
    """Render a line with ``%g`` (``-1.5``, ``2.5``, ``220.5``), normalizing ``-0.0``."""
    return f"{0.0 if line == 0 else line:g}"


@dataclass(frozen=True)
class _ParsedLeg:
    """A canonical leg key split into components; player fields only for PLAYER_PROP."""

    market: str
    side: str
    line: float | None = None
    player_id: str | None = None
    stat_key: str | None = None


def _parse_line(leg_key: str, raw: str) -> float:
    try:
        line = float(raw)
    except ValueError:
        raise UnknownLegError(f"Unknown leg key {leg_key!r}: line {raw!r} is not a number") from None
    if not math.isfinite(line):
        raise UnknownLegError(f"Unknown leg key {leg_key!r}: line must be finite")
    return line


def _parse_player_leg(leg_key: str, parts: list[str]) -> _ParsedLeg:
    """Parse ``PLAYER_PROP:{player}:{stat}:OVER|UNDER:{line}`` / ``...:YES|NO``.

    The side must match the stat's settlement type: YES/NO only on
    ``YES_NO_STAT_KEYS`` stats, OVER/UNDER (with a line) on everything else.
    """
    if len(parts) not in (4, 5) or not parts[1] or not parts[2]:
        raise UnknownLegError(f"Unknown leg key {leg_key!r}; {_LEG_FORMAT_HINT}")
    player_id, stat_key, side = parts[1], parts[2], parts[3]
    if len(parts) == 4:
        if side not in _PLAYER_YES_NO_SIDES:
            raise UnknownLegError(f"Unknown leg key {leg_key!r}; {_LEG_FORMAT_HINT}")
        if stat_key not in YES_NO_STAT_KEYS:
            raise UnknownLegError(
                f"Unknown leg key {leg_key!r}: {side} applies only to YES/NO stats "
                f"({', '.join(sorted(YES_NO_STAT_KEYS))}); use "
                f"'PLAYER_PROP:{player_id}:{stat_key}:OVER|UNDER:<line>' for {stat_key!r}"
            )
        return _ParsedLeg(_PLAYER_PROP_MARKET, side, None, player_id, stat_key)
    if side not in _PLAYER_OVER_UNDER_SIDES:
        raise UnknownLegError(f"Unknown leg key {leg_key!r}; {_LEG_FORMAT_HINT}")
    if stat_key in YES_NO_STAT_KEYS:
        raise UnknownLegError(
            f"Unknown leg key {leg_key!r}: {stat_key!r} settles YES/NO and takes no line; "
            f"use 'PLAYER_PROP:{player_id}:{stat_key}:YES'"
        )
    return _ParsedLeg(_PLAYER_PROP_MARKET, side, _parse_line(leg_key, parts[4]), player_id, stat_key)


def _parse_leg(leg_key: str) -> _ParsedLeg:
    """Split a canonical leg key into components; line is None for moneylines and YES/NO legs."""
    parts = leg_key.split(":")
    market = parts[0]
    if market == _PLAYER_PROP_MARKET:
        return _parse_player_leg(leg_key, parts)
    sides = _SIDES.get(market)
    if sides is None:
        raise UnknownLegError(f"Unknown leg key {leg_key!r}: unknown market {market!r}; {_LEG_FORMAT_HINT}")
    if market in _MARKETS_TWO_PART:
        if len(parts) != 2 or parts[1] not in sides:
            raise UnknownLegError(f"Unknown leg key {leg_key!r}; {_LEG_FORMAT_HINT}")
        return _ParsedLeg(market, parts[1])
    if len(parts) != 3 or parts[1] not in sides:
        raise UnknownLegError(f"Unknown leg key {leg_key!r}; {_LEG_FORMAT_HINT}")
    return _ParsedLeg(market, parts[1], _parse_line(leg_key, parts[2]))


def leg_vector(output: SimulationOutput, leg_key: str) -> npt.NDArray[np.bool_]:
    """Per-iteration boolean vector for one canonical leg key.

    Pushes are excluded (count as False): all comparisons are strict, so on an
    integer line the iterations landing exactly on the line satisfy neither
    side. Raises :class:`UnknownLegError` for keys outside the vocabulary,
    including player-prop legs whose player or stat was not captured.
    """
    parsed = _parse_leg(leg_key)
    market, side, line = parsed.market, parsed.side, parsed.line
    if market == _PLAYER_PROP_MARKET:
        return _player_leg_vector(output, leg_key, parsed)
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


def _player_leg_vector(output: SimulationOutput, leg_key: str, parsed: _ParsedLeg) -> npt.NDArray[np.bool_]:
    """Per-iteration boolean vector for one PLAYER_PROP leg (Phase 7 Wave 4)."""
    assert parsed.player_id is not None and parsed.stat_key is not None
    stats = output.player_stats.get(parsed.player_id)
    if stats is None:
        raise UnknownLegError(
            f"Unknown leg key {leg_key!r}: player {parsed.player_id!r} has no captured stats "
            "(run without include_player_props, or player absent from both rosters)"
        )
    values = stats.get(parsed.stat_key)
    if values is None:
        raise UnknownLegError(
            f"Unknown leg key {leg_key!r}: stat {parsed.stat_key!r} was not captured for player "
            f"{parsed.player_id!r} (captured: {', '.join(sorted(stats))})"
        )
    if parsed.side == "YES":
        return np.asarray(values > 0)
    if parsed.side == "NO":
        return np.asarray(values == 0)
    assert parsed.line is not None
    if parsed.side == "OVER":
        return np.asarray(values > parsed.line)
    return np.asarray(values < parsed.line)


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


def default_player_legs(
    output: SimulationOutput,
    line_cap: int = PLAYER_LEG_LINE_CAP,
    max_legs: int = MAX_PLAYER_LEGS,
) -> list[str]:
    """Default stored player legs when the run captured player props (Phase 7 Wave 4).

    Per player per stat (both sorted, so the order is deterministic):

    - YES_NO stats contribute the single YES leg (NO resolves as its exact
      complement at read time);
    - OVER_UNDER stats contribute OVER legs on the same half-point grid the
      player-distributions endpoint uses (:func:`default_line_grid` anchored
      at the stat mean), capped at ``line_cap`` lines closest to the mean.

    The total is capped at ``max_legs`` for artifact size control; truncation
    is logged at WARNING (never silent).
    """
    legs: list[str] = []
    for player_id in sorted(output.player_stats):
        for stat_key in sorted(output.player_stats[player_id]):
            if stat_key in YES_NO_STAT_KEYS:
                legs.append(f"PLAYER_PROP:{player_id}:{stat_key}:YES")
                continue
            mean = float(np.mean(output.player_stats[player_id][stat_key]))
            grid = sorted(default_line_grid(mean), key=lambda line: (abs(line - mean), line))
            lines = sorted(grid[:line_cap])
            legs.extend(f"PLAYER_PROP:{player_id}:{stat_key}:OVER:{format_line(line)}" for line in lines)
    if len(legs) > max_legs:
        logger.warning(
            "correlation artifact player legs capped at %d: dropping %d of %d candidate legs "
            "(deterministic order: sorted players, stats, lines)",
            max_legs,
            len(legs) - max_legs,
            len(legs),
        )
        legs = legs[:max_legs]
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
        Player-prop legs generalize identically: ``...:NO`` is the negation of
        the stored ``...:YES`` row, ``...:UNDER:{half-line}`` the negation of
        the stored ``...:OVER:{half-line}`` row.
        """
        index = self._leg_index().get(leg_key)
        if index is not None:
            return _LegRef(index, False)
        parsed = _parse_leg(leg_key)
        market, side, line = parsed.market, parsed.side, parsed.line
        complement: str | None = None
        if market == _PLAYER_PROP_MARKET:
            if not any(leg.startswith(f"{_PLAYER_PROP_MARKET}:") for leg in self.legs):
                raise UnknownLegError(
                    f"Leg {leg_key!r} is a player-prop leg but this run captured no player props; "
                    "re-run the simulation with include_player_props=true to store player legs"
                )
            if side == "NO":
                complement = f"PLAYER_PROP:{parsed.player_id}:{parsed.stat_key}:YES"
            elif side == "UNDER" and line is not None and not line.is_integer():
                complement = f"PLAYER_PROP:{parsed.player_id}:{parsed.stat_key}:OVER:{format_line(line)}"
        elif line is not None and not line.is_integer():
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

    When ``legs`` is None and the output carries captured player stats
    (Phase 7 Wave 4), the default vocabulary is extended with
    :func:`default_player_legs`, so mixed team+player subsets (same-game
    parlays like "home ML + anytime goalscorer") get exact empirical joints
    at read time. Team-only outputs produce artifacts byte-identical to the
    Wave 1 behavior.
    """
    leg_keys = default_legs(output, include_draw=include_draw) if legs is None else legs
    if legs is None and output.player_stats:
        leg_keys = [*leg_keys, *default_player_legs(output)]
    if not leg_keys:
        raise UnknownLegError("At least one leg is required to build a correlation artifact")
    vectors = np.stack([leg_vector(output, leg) for leg in leg_keys])
    marginals = {leg: round(float(np.mean(vec)), 6) for leg, vec in zip(leg_keys, vectors, strict=True)}
    packed = np.packbits(vectors, axis=1).tobytes()
    logger.debug(
        "correlation artifact: %d legs x %d iterations, packed matrix %d bytes",
        len(leg_keys),
        output.iterations_run,
        len(packed),
    )
    return CorrelationArtifact(
        legs=list(leg_keys),
        marginals=marginals,
        matrix=_correlation_matrix(vectors),
        iterations=output.iterations_run,
        packed_matrix=packed,
        joint_goal_grid=(
            [[round(float(v), 8) for v in row] for row in joint_goal_grid] if joint_goal_grid is not None else None
        ),
    )
