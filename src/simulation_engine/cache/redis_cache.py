"""Redis key layout for simulation results.

Keys per schemas/database-schemas/redis-schemas.md, plus three additions
required because this service has no Postgres (proposed as a docs update):

- ``sim:run:{simulation_run_id}`` -- full run record for GET /simulations/{id}
- ``sim:latest:{game_id}`` -- latest run id pointer for /games/{id}/latest
- ``sim:batch:{batch_id}`` / ``sim:idempotency:{key}`` / daily load counter

Distributions — and the ``sim:correlations:{game_id}`` parlay-correlation
artifact (Phase 7 Wave 1) — are stored zlib-compressed then base64-encoded so
a single ``decode_responses=True`` client can be used throughout.

``sim:distributions:{game_id}``, ``sim:correlations:{game_id}``, and
``sim:player_distributions:{game_id}`` (Phase 7 Wave 3) are game-scoped
LATEST-RUN blobs by design: each stores the run id it belongs to, and the
read path 404s when the requested run is no longer the latest. Live
re-simulations (Phase 7 Wave 2) therefore overwrite a game's pregame blobs
with latest-wins semantics — never a collision, because run reuse is keyed by
``sim:result:{game_id}:{parameters_hash}`` and live runs carry a distinct
parameters hash. The player blob is written only by props-enabled runs; a
later props-off run leaves the stale blob in place, and the run-id check on
the read path turns it into a 404 rather than serving mismatched data.
"""

import base64
import json
import zlib
from collections.abc import Mapping
from datetime import UTC, datetime

import redis.asyncio as aioredis
from redis.typing import EncodableT, FieldT

from simulation_engine.api.models import SimulationRunData


class SimulationCache:
    def __init__(self, redis_client: "aioredis.Redis", result_ttl: int, idempotency_ttl: int) -> None:
        self._redis = redis_client
        self._result_ttl = result_ttl
        self._idempotency_ttl = idempotency_ttl

    @staticmethod
    def _result_key(game_id: str, parameters_hash: str) -> str:
        return f"sim:result:{game_id}:{parameters_hash}"

    async def get_cached_run_id(self, game_id: str, parameters_hash: str) -> str | None:
        run_id = await self._redis.hget(self._result_key(game_id, parameters_hash), "simulation_run_id")
        return str(run_id) if run_id else None

    async def get_run(self, run_id: str) -> SimulationRunData | None:
        raw = await self._redis.get(f"sim:run:{run_id}")
        if raw is None:
            return None
        return SimulationRunData.model_validate_json(raw)

    async def get_latest_run_id(self, game_id: str) -> str | None:
        run_id = await self._redis.get(f"sim:latest:{game_id}")
        return str(run_id) if run_id else None

    async def get_distributions(self, game_id: str) -> dict[str, object] | None:
        raw = await self._redis.get(f"sim:distributions:{game_id}")
        if raw is None:
            return None
        decoded: dict[str, object] = json.loads(zlib.decompress(base64.b64decode(raw)))
        return decoded

    async def get_correlations(self, game_id: str) -> dict[str, object] | None:
        raw = await self._redis.get(f"sim:correlations:{game_id}")
        if raw is None:
            return None
        decoded: dict[str, object] = json.loads(zlib.decompress(base64.b64decode(raw)))
        return decoded

    async def get_player_distributions(self, game_id: str) -> dict[str, object] | None:
        raw = await self._redis.get(f"sim:player_distributions:{game_id}")
        if raw is None:
            return None
        decoded: dict[str, object] = json.loads(zlib.decompress(base64.b64decode(raw)))
        return decoded

    @staticmethod
    def _compress_blob(run_id: str, payload: Mapping[str, object]) -> str:
        return base64.b64encode(zlib.compress(json.dumps({"simulation_run_id": run_id, **payload}).encode())).decode()

    async def store_run(
        self,
        run: SimulationRunData,
        distributions: Mapping[str, object],
        correlations: Mapping[str, object],
        player_distributions: Mapping[str, object] | None = None,
    ) -> None:
        """Persist all read paths for a completed run in one pipeline.

        ``player_distributions`` is written only when the run captured player
        props (Phase 7 Wave 3); None (props off) leaves any previous game
        blob untouched — its embedded run id no longer matches the latest
        run, so the read path 404s instead of serving stale data.
        """
        result_fields: dict[FieldT, EncodableT] = {
            "simulation_run_id": run.simulation_run_id,
            "home_win_prob": run.result.home_win_probability,
            "away_win_prob": run.result.away_win_probability,
            "mean_home_score": run.result.mean_home_score,
            "mean_away_score": run.result.mean_away_score,
            "mean_total": run.result.mean_total,
            "mean_margin": run.result.mean_margin,
            "spread_covers_json": json.dumps(run.result.spread_cover_probabilities),
            "total_overs_json": json.dumps(run.result.total_over_probabilities),
            "iterations": run.iterations_completed,
            "converged": int(run.converged),
            "completed_at": run.completed_at,
        }
        distributions_blob = self._compress_blob(run.simulation_run_id, distributions)
        correlations_blob = self._compress_blob(run.simulation_run_id, correlations)

        pipe = self._redis.pipeline(transaction=False)
        result_key = self._result_key(run.game_id, run.parameters_hash)
        pipe.hset(result_key, mapping=result_fields)
        pipe.expire(result_key, self._result_ttl)
        pipe.set(f"sim:run:{run.simulation_run_id}", run.model_dump_json(), ex=self._result_ttl)
        pipe.set(f"sim:latest:{run.game_id}", run.simulation_run_id, ex=self._result_ttl)
        pipe.set(f"sim:distributions:{run.game_id}", distributions_blob, ex=self._result_ttl)
        pipe.set(f"sim:correlations:{run.game_id}", correlations_blob, ex=self._result_ttl)
        if player_distributions is not None:
            player_blob = self._compress_blob(run.simulation_run_id, player_distributions)
            pipe.set(f"sim:player_distributions:{run.game_id}", player_blob, ex=self._result_ttl)
        pipe.incr(self._daily_counter_key())
        pipe.expire(self._daily_counter_key(), 172_800)
        await pipe.execute()

    @staticmethod
    def _daily_counter_key() -> str:
        return f"sim:load:simulations:{datetime.now(tz=UTC).date().isoformat()}"

    async def simulations_today(self) -> int:
        value = await self._redis.get(self._daily_counter_key())
        return int(value) if value else 0

    async def get_idempotent(self, key: str) -> tuple[str, str] | None:
        """Return (body_hash, run_id) recorded for an idempotency key."""
        raw = await self._redis.get(f"sim:idempotency:{key}")
        if raw is None:
            return None
        record = json.loads(raw)
        return str(record["body_hash"]), str(record["run_id"])

    async def store_idempotent(self, key: str, body_hash: str, run_id: str) -> None:
        await self._redis.set(
            f"sim:idempotency:{key}",
            json.dumps({"body_hash": body_hash, "run_id": run_id}),
            ex=self._idempotency_ttl,
        )

    async def store_batch(self, batch_id: str, payload: str) -> None:
        await self._redis.set(f"sim:batch:{batch_id}", payload, ex=self._result_ttl)

    async def is_healthy(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001 - any redis failure means unhealthy
            return False
