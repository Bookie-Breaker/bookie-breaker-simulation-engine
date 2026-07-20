"""Request/response models mirroring api-contracts/simulation-engine-api.md."""

from typing import Literal

from pydantic import BaseModel, Field

DistributionType = Literal["margin", "total", "home_score", "away_score", "all"]


class SimulationConfigIn(BaseModel):
    iterations: int = Field(default=10_000, ge=100, le=50_000)
    convergence_threshold: float = Field(default=0.005, gt=0.0)
    random_seed: int | None = None
    plugin_config: dict[str, object] = Field(default_factory=dict)
    # Phase 7 Wave 3: opt into the detailed path that captures per-player
    # stat distributions (soccer + basketball in v1). False keeps team-level
    # output byte-identical to pre-Wave-3 behavior; the service strips the
    # False value from hashed payloads so pregame parameter hashes and
    # idempotency body hashes are unchanged.
    include_player_props: bool = False


class LiveStateIn(BaseModel):
    """Current game state for live re-simulation (Phase 7 Wave 2).

    Simulates the remainder of the game and adds the current score as an
    offset. Bounds (fraction_remaining in (0, 1], scores >= 0, sport-specific
    refinements in their legal ranges) are enforced in the service layer as
    422 UNPROCESSABLE_ENTITY per the Wave 2 contract — deliberately not as
    pydantic Field constraints, which this service's RequestValidationError
    handler would surface as 400 VALIDATION_ERROR instead.

    Sport-specific optional fields: ``bases``/``outs``/``half`` (baseball;
    bases is a 3-char occupancy string like "1-3", half is "TOP"/"BOTTOM",
    period is the inning number), ``possession``/``down``/``yardline``
    (football; possession is "HOME"/"AWAY").
    """

    home_score: int
    away_score: int
    fraction_remaining: float
    period: int | None = None
    clock_seconds: int | None = None
    bases: str | None = None
    outs: int | None = None
    half: str | None = None
    possession: str | None = None
    down: int | None = None
    yardline: int | None = None


class SimulationRequest(BaseModel):
    game_id: str
    config: SimulationConfigIn = SimulationConfigIn()
    force_refresh: bool = False
    live_state: LiveStateIn | None = None


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
    # live_state is accepted by the schema but rejected with 422 by the
    # service: live re-simulation is single-game only in v1 (Phase 7 Wave 2).
    game_id: str
    config: SimulationConfigIn | None = None
    live_state: LiveStateIn | None = None


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


class PlayerStatBlock(BaseModel):
    """Distribution plus market-facing probabilities for one player stat (Phase 7 Wave 3).

    OVER_UNDER stats carry ``over_probabilities`` — P(count > line) for
    half-point lines around the mean (monotonically non-increasing in the
    line). YES_NO stats (``player_goal_scorer_anytime``, ``player_anytime_td``)
    carry ``yes_probability`` = P(count > 0) instead of a line grid.
    """

    distribution: Distribution
    over_probabilities: dict[str, float] | None = None
    yes_probability: float | None = None


class PlayerPropsEntry(BaseModel):
    """One player's captured stats keyed by canonical stat key (ADR-029)."""

    name: str
    team: Literal["HOME", "AWAY"]
    stats: dict[str, PlayerStatBlock]


class PlayerDistributionsData(BaseModel):
    """Per-player stat distributions for one simulation run (Phase 7 Wave 3).

    ``players`` is keyed by statistics-service player UUID. Stat keys are the
    canonical Odds API market keys, so downstream services join these to
    market lines without translation. Empty when props were requested but no
    roster data existed (dormant sports, empty upstream rosters).
    """

    simulation_run_id: str
    game_id: str
    iterations_completed: int
    players: dict[str, PlayerPropsEntry]


class CorrelationsData(BaseModel):
    """Same-game parlay correlation artifact for one simulation run (Phase 7 Wave 1).

    ``legs`` uses the canonical leg vocabulary (``MONEYLINE:HOME``,
    ``SPREAD:HOME:-1.5``, ``TOTAL:OVER:2.5``, ...; lines rendered with %g).
    ``matrix`` is the pairwise phi/Pearson correlation matrix aligned with
    ``legs`` (unit diagonal; zero-variance legs correlate 0.0 with everything).
    ``joint_probability`` is present only when specific legs were requested:
    the empirical Monte Carlo probability that ALL requested legs hit in the
    same iteration (pushes count as misses). ``joint_goal_grid`` is the
    analytic joint score PMF (rows = home score) for Poisson-grid sports
    (soccer regulation, hockey pre-OT regulation); null elsewhere.
    """

    simulation_run_id: str
    game_id: str
    iterations: int
    legs: list[str]
    marginals: dict[str, float]
    matrix: list[list[float]]
    joint_probability: float | None = None
    joint_goal_grid: list[list[float]] | None = None


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
