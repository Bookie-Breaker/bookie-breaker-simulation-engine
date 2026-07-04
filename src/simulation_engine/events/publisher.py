"""Publish simulation.completed events per redis-schemas.md.

Events are fire-and-forget: publish failures are logged and never fail the
simulation request (pub/sub is non-critical per communication-patterns).
"""

import json
import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis

from simulation_engine.api.models import SimulationRunData

logger = logging.getLogger(__name__)

CHANNEL = "events:simulation.completed"


async def publish_simulation_completed(redis_client: "aioredis.Redis", run: SimulationRunData, league: str) -> None:
    payload = {
        "event": "simulation.completed",
        "timestamp": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "simulation_run_id": run.simulation_run_id,
        "game_id": run.game_id,
        "league": league,
        "iterations": run.iterations_completed,
        "converged": run.converged,
        "duration_ms": run.duration_ms,
        "home_win_prob": run.result.home_win_probability,
        "mean_margin": run.result.mean_margin,
    }
    try:
        await redis_client.publish(CHANNEL, json.dumps(payload))
    except Exception:  # noqa: BLE001 - pub/sub is best-effort by design
        logger.warning("failed to publish %s for run %s", CHANNEL, run.simulation_run_id, exc_info=True)
