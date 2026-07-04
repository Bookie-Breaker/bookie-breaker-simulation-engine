"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI

from simulation_engine import __version__
from simulation_engine.api.envelope import RequestIDMiddleware
from simulation_engine.api.errors import register_error_handlers
from simulation_engine.api.routes import games, health, simulations
from simulation_engine.cache.redis_cache import SimulationCache
from simulation_engine.clients.statistics import StatisticsClient
from simulation_engine.config import Settings, get_settings
from simulation_engine.services.simulation_service import SimulationService
from simulation_engine.telemetry import configure_telemetry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        redis_client: aioredis.Redis = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        statistics = StatisticsClient(settings.statistics_service_url, http_client)
        cache = SimulationCache(redis_client, settings.result_ttl_seconds, settings.idempotency_ttl_seconds)
        app.state.simulation_service = SimulationService(settings, cache, statistics, redis_client)
        try:
            yield
        finally:
            await http_client.aclose()
            await redis_client.aclose()

    app = FastAPI(
        title="BookieBreaker Simulation Engine",
        version=__version__,
        description="Monte Carlo simulations for sports outcome distributions.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)
    app.include_router(simulations.router, prefix="/api/v1/sim")
    app.include_router(games.router, prefix="/api/v1/sim")
    app.include_router(health.router, prefix="/api/v1/sim")
    configure_telemetry(app, settings)
    return app


app = create_app()
