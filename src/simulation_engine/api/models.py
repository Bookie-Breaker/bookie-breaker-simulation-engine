"""Request/response models mirroring api-contracts/simulation-engine-api.md."""

from typing import Literal

from pydantic import BaseModel, Field

DistributionType = Literal["margin", "total", "home_score", "away_score", "all"]


class SimulationConfigIn(BaseModel):
    iterations: int = Field(default=10_000, ge=100, le=50_000)
    convergence_threshold: float = Field(default=0.005, gt=0.0)
    random_seed: int | None = None
    plugin_config: dict[str, object] = Field(default_factory=dict)


class SimulationRequest(BaseModel):
    game_id: str
    config: SimulationConfigIn = SimulationConfigIn()
    force_refresh: bool = False


class SimulationConfigOut(BaseModel):
    sport: str
    iterations: int
    convergence_threshold: float
    random_seed: int | None = None


class Percentiles(BaseModel):
    margin: dict[str, int]
    total: dict[str, int]


class SimulationResultData(BaseModel):
    """Aggregated simulation result.

    ``spread_push_probabilities`` / ``total_push_probabilities`` carry
    P(margin == line) / P(total == line) for INTEGER lines only — half-point
    lines cannot push and are omitted rather than serialized as 0.0. Cover
    and over probabilities are unchanged: strictly-greater-than semantics.
    """

    id: str
    home_win_probability: float
    away_win_probability: float
    draw_probability: float
    mean_home_score: float
    mean_away_score: float
    mean_total: float
    mean_margin: float
    spread_cover_probabilities: dict[str, float]
    total_over_probabilities: dict[str, float]
    spread_push_probabilities: dict[str, float] = Field(default_factory=dict)
    total_push_probabilities: dict[str, float] = Field(default_factory=dict)
    percentiles: Percentiles


class SimulationRunData(BaseModel):
    simulation_run_id: str
    game_id: str
    status: str
    cached: bool = False
    config: SimulationConfigOut
    started_at: str
    completed_at: str
    duration_ms: int
    iterations_completed: int
    converged: bool
    parameters_hash: str
    batch_id: str | None = None
    result: SimulationResultData


class BatchGameRequest(BaseModel):
    game_id: str
    config: SimulationConfigIn | None = None


class BatchRequest(BaseModel):
    games: list[BatchGameRequest] = Field(min_length=1, max_length=50)
    default_config: SimulationConfigIn = SimulationConfigIn()
    force_refresh: bool = False


class BatchResultSummary(BaseModel):
    home_win_probability: float
    away_win_probability: float
    mean_total: float
    mean_margin: float


class BatchGameResult(BaseModel):
    simulation_run_id: str | None = None
    game_id: str
    status: str
    cached: bool = False
    result: BatchResultSummary | None = None
    error: str | None = None


class BatchData(BaseModel):
    batch_id: str
    status: Literal["completed", "partial", "failed"]
    total_games: int
    completed_games: int
    failed_games: int
    started_at: str
    completed_at: str
    total_duration_ms: int
    results: list[BatchGameResult]


class Distribution(BaseModel):
    type: str = "discrete"
    values: dict[str, float]
    mean: float
    std_dev: float
    min: int
    max: int


class DistributionsData(BaseModel):
    simulation_run_id: str
    game_id: str
    iterations_completed: int
    distributions: dict[str, Distribution]


class HealthLoad(BaseModel):
    active_simulations: int
    queued_simulations: int
    max_concurrent: int
    simulations_today: int
    avg_duration_ms: int


class HealthData(BaseModel):
    status: str
    service: str = "simulation-engine"
    version: str
    uptime_seconds: int
    dependencies: dict[str, str]
    load: HealthLoad
