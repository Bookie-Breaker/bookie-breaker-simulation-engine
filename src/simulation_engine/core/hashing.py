"""Stable parameter hashing for the simulation result cache.

The hash covers everything that determines a simulation's output: the game,
both teams' parameters, game context, the effective config, and an engine
identity (the plugin label plus a version). Callers pass the registry's
``PluginSpec.label`` as ``plugin_label`` so different sports never collide in
the cache; the default preserves pre-Phase-6 NBA hashes byte-for-byte. Bump
ENGINE_VERSION whenever plugin math changes so stale cached results are never
served across deployments.
"""

import hashlib
import json
from dataclasses import asdict
from typing import Any

from simulation_engine.core.params import GameContext, PlayerRates, SportParams

ENGINE_VERSION = 1
#: Version of the player-prop allocation models (Phase 7 Wave 3). Folded into
#: the canonical payload ONLY when the context carries a roster_signature
#: (i.e. player capture participates in the run), so pregame digests are
#: byte-identical to pre-Wave-3 digests. Bump when any per-sport allocation
#: model changes so stale cached player distributions are never served.
PROP_ENGINE_VERSION = 1
HASH_LENGTH = 12
ROSTER_SIGNATURE_LENGTH = 16


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonicalize(v) for v in value]
    return value


def _strip_none_fields(value: Any) -> Any:
    """Recursively drop None-valued dict entries (context canonicalization).

    Applied to the context only: an optional field that is unset must hash
    exactly like a context from before the field existed, at every nesting
    level — a LiveState with unset sport-specific fields (asdict turns the
    nested dataclass into a dict) canonicalizes to the same digest as one
    that never mentions them.
    """
    if isinstance(value, dict):
        return {k: _strip_none_fields(v) for k, v in value.items() if v is not None}
    return value


def compute_roster_signature(home: list[PlayerRates], away: list[PlayerRates]) -> str:
    """Stable SHA over the sorted PlayerRates inputs (Phase 7 Wave 3).

    Sorted by player_id within each side so provider ordering never changes
    the signature; floats are canonicalized like every other hash input. The
    signature goes into ``GameContext.roster_signature`` and thus the
    parameters hash: any roster or rate change invalidates cached
    props-enabled simulations, while props-off runs (signature None) keep
    their pregame digests.
    """
    payload = _canonicalize(
        {
            "home": [asdict(p) for p in sorted(home, key=lambda p: p.player_id)],
            "away": [asdict(p) for p in sorted(away, key=lambda p: p.player_id)],
        }
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:ROSTER_SIGNATURE_LENGTH]


def compute_parameters_hash(
    game_id: str,
    home_params: SportParams,
    away_params: SportParams,
    context: GameContext,
    config: dict[str, Any],
    plugin_label: str = "basketball",
) -> str:
    engine: dict[str, Any] = {"plugin": plugin_label, "version": ENGINE_VERSION}
    if context.roster_signature is not None:
        # Player capture participates in this run: fold the prop-model
        # version in so allocation-model changes invalidate cached player
        # distributions. Absent when props are off, keeping pregame payloads
        # (and the pinned pregame digest) byte-identical to pre-Wave-3.
        engine["prop_version"] = PROP_ENGINE_VERSION
    payload = _canonicalize(
        {
            "game_id": game_id,
            "home_params": asdict(home_params),
            "away_params": asdict(away_params),
            # None-valued context fields (e.g. unannounced probable starters,
            # pregame live_state, unset LiveState refinements) are stripped
            # recursively so hashes computed before a field existed stay
            # byte-identical; setting a field changes the hash and invalidates
            # cached simulations, which is the desired behavior.
            "context": _strip_none_fields(asdict(context)),
            "config": config,
            "engine": engine,
        }
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:HASH_LENGTH]
