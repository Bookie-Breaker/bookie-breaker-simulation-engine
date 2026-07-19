"""Simulation orchestration: fetch stats, cache-check, run, persist, publish."""

import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from datetime import UTC, datetime

import redis.asyncio as aioredis

from simulation_engine import __version__
from simulation_engine.api.errors import ApiError, DuplicateResourceError, NotFoundError, UnprocessableError
from simulation_engine.api.models import (
    BatchData,
    BatchGameRequest,
    BatchGameResult,
    BatchResultSummary,
    CorrelationsData,
    DistributionsData,
    DistributionType,
    HealthData,
    HealthLoad,
    SimulationConfigIn,
    SimulationConfigOut,
    SimulationRunData,
)
from simulation_engine.cache.redis_cache import SimulationCache
from simulation_engine.clients.statistics import ProbablePitcher, StatisticsClient
from simulation_engine.config import Settings
from simulation_engine.core.correlations import CorrelationArtifact, UnknownLegError, build_correlation_artifact
from simulation_engine.core.hashing import compute_parameters_hash
from simulation_engine.core.output import build_distributions, build_result
from simulation_engine.core.params import GameContext
from simulation_engine.core.plugins import get_plugin
from simulation_engine.core.runner import run_monte_carlo
from simulation_engine.events.publisher import publish_simulation_completed

_TERMINAL_GAME_STATUSES = frozenset({"FINAL", "CANCELLED"})


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _starter_fip(pitcher: ProbablePitcher | None) -> float | None:
    """FIP of an announced probable starter; None when unannounced or FIP is missing."""
    if pitcher is None or pitcher.fip <= 0:
        return None
    return pitcher.fip


def _request_body_hash(game_id: str, config: SimulationConfigIn, force_refresh: bool) -> str:
    canonical = json.dumps(
        {"game_id": game_id, "config": config.model_dump(mode="json"), "force_refresh": force_refresh},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class SimulationService:
    def __init__(
        self,
        settings: Settings,
        cache: SimulationCache,
        statistics: StatisticsClient,
        redis_client: "aioredis.Redis",
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._statistics = statistics
        self._redis = redis_client
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_simulations)
        self._active = 0
        self._queued = 0
        self._recent_durations: deque[int] = deque(maxlen=100)
        self._started_monotonic = time.monotonic()

    async def run_simulation(
        self,
        game_id: str,
        config: SimulationConfigIn,
        force_refresh: bool = False,
        idempotency_key: str | None = None,
    ) -> SimulationRunData:
        body_hash = _request_body_hash(game_id, config, force_refresh)
        if idempotency_key is not None:
            existing = await self._cache.get_idempotent(idempotency_key)
            if existing is not None:
                stored_hash, stored_run_id = existing
                if stored_hash != body_hash:
                    raise DuplicateResourceError("X-Idempotency-Key was already used with a different request body")
                replayed = await self._cache.get_run(stored_run_id)
                if replayed is not None:
                    return replayed.model_copy(update={"cached": True})

        game = await self._statistics.get_game(game_id)
        if game.status in _TERMINAL_GAME_STATUSES:
            raise UnprocessableError(f"Game {game_id} is {game.status}; simulation is not applicable")

        spec = get_plugin(game.league)
        home_stats, away_stats = await asyncio.gather(
            self._statistics.get_team_stats(game.home_team.id),
            self._statistics.get_team_stats(game.away_team.id),
        )
        home_params = spec.map_team_stats(home_stats)
        away_params = spec.map_team_stats(away_stats)
        # Probable starters (BASEBALL leagues) enter the context — and thus
        # the parameters hash — so a starter announcement invalidates cached
        # simulations. The baseball plugin applies each starter to the
        # OPPOSING batting side (home batters face the away starter).
        context = GameContext(
            league=game.league,
            home_starter_fip=_starter_fip(game.home_probable_pitcher),
            away_starter_fip=_starter_fip(game.away_probable_pitcher),
        )
        config_dict = config.model_dump(mode="json")
        parameters_hash = compute_parameters_hash(
            game_id, home_params, away_params, context, config_dict, plugin_label=spec.label
        )

        if not force_refresh:
            cached_run_id = await self._cache.get_cached_run_id(game_id, parameters_hash)
            if cached_run_id is not None:
                cached_run = await self._cache.get_run(cached_run_id)
                if cached_run is not None:
                    return cached_run.model_copy(update={"cached": True})

        simulator = spec.simulator({**spec.plugin_config, **config.plugin_config})
        started_at = _utc_now_iso()
        self._queued += 1
        async with self._semaphore:
            self._queued -= 1
            self._active += 1
            try:
                output = await asyncio.to_thread(
                    run_monte_carlo,
                    simulator,
                    home_params,
                    away_params,
                    context,
                    config.iterations,
                    config.convergence_threshold,
                    self._settings.convergence_check_interval,
                    config.random_seed,
                    grid_config=spec.grid_config,
                )
            finally:
                self._active -= 1

        duration_ms = int(output.elapsed_ms)
        self._recent_durations.append(duration_ms)
        run = SimulationRunData(
            simulation_run_id=str(uuid.uuid4()),
            game_id=game_id,
            status="completed",
            cached=False,
            config=SimulationConfigOut(
                sport=simulator.get_sport(),
                iterations=config.iterations,
                convergence_threshold=config.convergence_threshold,
                random_seed=config.random_seed,
            ),
            started_at=started_at,
            completed_at=_utc_now_iso(),
            duration_ms=duration_ms,
            iterations_completed=output.iterations_run,
            converged=output.converged,
            parameters_hash=parameters_hash,
            result=build_result(output, str(uuid.uuid4())),
        )

        distributions = {name: dist.model_dump(mode="json") for name, dist in build_distributions(output).items()}
        # The raw per-iteration sample arrays are not retained past this point,
        # so the parlay-correlation artifact (including the packed boolean leg
        # matrix that makes arbitrary-subset joints computable at read time)
        # must be built here, from the in-memory output.
        correlations = build_correlation_artifact(
            output,
            include_draw=simulator.get_sport() == "SOCCER",
            joint_goal_grid=simulator.joint_grid(),
        ).to_payload()
        await self._cache.store_run(run, distributions, correlations)
        await publish_simulation_completed(self._redis, run, game.league)
        if idempotency_key is not None:
            await self._cache.store_idempotent(idempotency_key, body_hash, run.simulation_run_id)
        return run

    async def get_run(self, simulation_id: str) -> SimulationRunData:
        run = await self._cache.get_run(simulation_id)
        if run is None:
            raise NotFoundError(f"Simulation run {simulation_id} not found (results expire after 2 hours)")
        return run

    async def get_latest(self, game_id: str, force_refresh: bool = False) -> SimulationRunData:
        if force_refresh:
            return await self.run_simulation(game_id, self.default_config(), force_refresh=True)
        run_id = await self._cache.get_latest_run_id(game_id)
        if run_id is None:
            raise NotFoundError(f"No simulations found for game {game_id}")
        return await self.get_run(run_id)

    async def get_distributions(self, simulation_id: str, distribution_type: DistributionType) -> DistributionsData:
        run = await self.get_run(simulation_id)
        stored = await self._cache.get_distributions(run.game_id)
        if stored is None or stored.get("simulation_run_id") != simulation_id:
            raise NotFoundError(f"Distributions for simulation {simulation_id} are no longer available")
        names = ["home_score", "away_score", "margin", "total"] if distribution_type == "all" else [distribution_type]
        distributions = {name: stored[name] for name in names if name in stored}
        return DistributionsData.model_validate(
            {
                "simulation_run_id": simulation_id,
                "game_id": run.game_id,
                "iterations_completed": run.iterations_completed,
                "distributions": distributions,
            }
        )

    async def get_correlations(self, simulation_id: str, legs: list[str] | None = None) -> CorrelationsData:
        """Correlation artifact for a run; with ``legs``, a subset view plus their joint probability.

        Requested legs must be stored artifact legs or exact half-point
        complements — the raw sample arrays are gone after the run, so legs
        outside that vocabulary cannot be recomputed and yield a 422.
        """
        run = await self.get_run(simulation_id)
        stored = await self._cache.get_correlations(run.game_id)
        if stored is None or stored.get("simulation_run_id") != simulation_id:
            raise NotFoundError(f"Correlations for simulation {simulation_id} are no longer available")
        artifact = CorrelationArtifact.from_payload(stored)
        if legs is None:
            return CorrelationsData(
                simulation_run_id=simulation_id,
                game_id=run.game_id,
                iterations=artifact.iterations,
                legs=artifact.legs,
                marginals=artifact.marginals,
                matrix=artifact.matrix,
                joint_goal_grid=artifact.joint_goal_grid,
            )
        try:
            marginals, matrix, joint = artifact.subset(legs)
        except UnknownLegError as exc:
            raise UnprocessableError(str(exc)) from exc
        return CorrelationsData(
            simulation_run_id=simulation_id,
            game_id=run.game_id,
            iterations=artifact.iterations,
            legs=legs,
            marginals=marginals,
            matrix=matrix,
            joint_probability=joint,
            joint_goal_grid=artifact.joint_goal_grid,
        )

    def default_config(self) -> SimulationConfigIn:
        return SimulationConfigIn(
            iterations=min(self._settings.simulation_iterations, self._settings.max_iterations),
            convergence_threshold=self._settings.convergence_threshold,
        )

    async def run_batch(
        self,
        games: list[BatchGameRequest],
        default_config: SimulationConfigIn,
        force_refresh: bool = False,
    ) -> BatchData:
        batch_id = str(uuid.uuid4())
        started_at = _utc_now_iso()
        started = time.perf_counter()

        async def run_one(entry: BatchGameRequest) -> BatchGameResult:
            try:
                run = await self.run_simulation(entry.game_id, entry.config or default_config, force_refresh)
            except ApiError as exc:
                return BatchGameResult(game_id=entry.game_id, status="failed", error=exc.message)
            return BatchGameResult(
                simulation_run_id=run.simulation_run_id,
                game_id=entry.game_id,
                status="completed",
                cached=run.cached,
                result=BatchResultSummary(
                    home_win_probability=run.result.home_win_probability,
                    away_win_probability=run.result.away_win_probability,
                    mean_total=run.result.mean_total,
                    mean_margin=run.result.mean_margin,
                ),
            )

        results = list(await asyncio.gather(*(run_one(entry) for entry in games)))
        failed = sum(1 for r in results if r.status == "failed")
        completed = len(results) - failed
        status: str = "completed" if failed == 0 else ("failed" if completed == 0 else "partial")
        batch = BatchData(
            batch_id=batch_id,
            status=status,  # type: ignore[arg-type]
            total_games=len(results),
            completed_games=completed,
            failed_games=failed,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            total_duration_ms=int((time.perf_counter() - started) * 1000),
            results=results,
        )
        await self._cache.store_batch(batch_id, batch.model_dump_json())
        return batch

    async def health(self) -> HealthData:
        redis_ok, stats_ok = await asyncio.gather(self._cache.is_healthy(), self._statistics.is_healthy())
        durations = list(self._recent_durations)
        return HealthData(
            status="healthy" if (redis_ok and stats_ok) else "degraded",
            version=__version__,
            uptime_seconds=int(time.monotonic() - self._started_monotonic),
            dependencies={
                "statistics_service": "healthy" if stats_ok else "unhealthy",
                "redis": "healthy" if redis_ok else "unhealthy",
            },
            load=HealthLoad(
                active_simulations=self._active,
                queued_simulations=self._queued,
                max_concurrent=self._settings.max_concurrent_simulations,
                simulations_today=await self._cache.simulations_today() if redis_ok else 0,
                avg_duration_ms=int(sum(durations) / len(durations)) if durations else 0,
            ),
        )
