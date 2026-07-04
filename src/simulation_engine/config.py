"""Runtime configuration via environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    port: int = 8003
    log_level: str = "info"
    redis_url: str = "redis://localhost:6379"
    statistics_service_url: str = "http://localhost:8002"

    simulation_iterations: int = 10_000
    max_iterations: int = 50_000
    convergence_threshold: float = 0.005
    convergence_check_interval: int = 1_000
    max_concurrent_simulations: int = 30

    result_ttl_seconds: int = 7_200  # 2h, per redis-schemas.md
    idempotency_ttl_seconds: int = 86_400  # 24h, per api-contracts README

    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "simulation-engine"


@lru_cache
def get_settings() -> Settings:
    return Settings()
