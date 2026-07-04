"""Stable parameter hashing for the simulation result cache.

The hash covers everything that determines a simulation's output: the game,
both teams' parameters, game context, the effective config, and an engine
version. Bump ENGINE_VERSION whenever plugin math changes so stale cached
results are never served across deployments.
"""

import hashlib
import json
from dataclasses import asdict
from typing import Any

from simulation_engine.core.params import GameContext, TeamParams

ENGINE_VERSION = 1
HASH_LENGTH = 12


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonicalize(v) for v in value]
    return value


def compute_parameters_hash(
    game_id: str,
    home_params: TeamParams,
    away_params: TeamParams,
    context: GameContext,
    config: dict[str, Any],
) -> str:
    payload = _canonicalize(
        {
            "game_id": game_id,
            "home_params": asdict(home_params),
            "away_params": asdict(away_params),
            "context": asdict(context),
            "config": config,
            "engine": {"plugin": "basketball", "version": ENGINE_VERSION},
        }
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:HASH_LENGTH]
