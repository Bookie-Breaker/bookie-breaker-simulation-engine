"""Simulation endpoints per api-contracts/simulation-engine-api.md."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Query

from simulation_engine.api.dependencies import get_simulation_service
from simulation_engine.api.envelope import Envelope, envelope
from simulation_engine.api.models import (
    BatchData,
    BatchRequest,
    CorrelationsData,
    DistributionsData,
    DistributionType,
    PlayerDistributionsData,
    SimulationRequest,
    SimulationRunData,
)
from simulation_engine.services.simulation_service import SimulationService

router = APIRouter(tags=["simulations"])

ServiceDep = Annotated[SimulationService, Depends(get_simulation_service)]
SimulationIdPath = Annotated[str, Path(description="The simulation run identifier.")]


@router.post("/simulations", status_code=201, response_model=Envelope[SimulationRunData])
async def create_simulation(
    request: SimulationRequest,
    service: ServiceDep,
    x_idempotency_key: Annotated[
        str | None, Header(description="UUID for idempotent submission (replayed for 24 hours).")
    ] = None,
) -> Envelope[SimulationRunData]:
    """Run a Monte Carlo simulation for a game.

    Returns a cached result when an identical parameters_hash exists and
    force_refresh is false. With ``live_state`` (Phase 7 Wave 2) the run
    simulates the remainder of the game from the given in-game state; live
    runs get their own parameters_hash, so they never collide with pregame
    cache entries.
    """
    run = await service.run_simulation(
        request.game_id,
        request.config,
        force_refresh=request.force_refresh,
        idempotency_key=x_idempotency_key,
        live_state=request.live_state,
    )
    return envelope(run)


@router.post("/simulations/batch", status_code=201, response_model=Envelope[BatchData])
async def create_batch(request: BatchRequest, service: ServiceDep) -> Envelope[BatchData]:
    """Simulate multiple games in parallel.

    Per-game failures do not fail the batch; the batch status is
    completed, partial, or failed.
    """
    batch = await service.run_batch(request.games, request.default_config, request.force_refresh)
    return envelope(batch)


@router.get("/simulations/{simulation_id}", response_model=Envelope[SimulationRunData])
async def get_simulation(simulation_id: SimulationIdPath, service: ServiceDep) -> Envelope[SimulationRunData]:
    """Get the results of a specific simulation run (results expire after 2 hours)."""
    return envelope(await service.get_run(simulation_id))


@router.get("/simulations/{simulation_id}/distributions", response_model=Envelope[DistributionsData])
async def get_simulation_distributions(
    simulation_id: SimulationIdPath,
    service: ServiceDep,
    distribution_type: Annotated[
        DistributionType, Query(description="Which distributions to return; 'all' returns every distribution.")
    ] = "all",
) -> Envelope[DistributionsData]:
    """Get raw score/margin/total distributions for a simulation, suitable for visualization."""
    return envelope(await service.get_distributions(simulation_id, distribution_type))


@router.get("/simulations/{simulation_id}/player-distributions", response_model=Envelope[PlayerDistributionsData])
async def get_simulation_player_distributions(
    simulation_id: SimulationIdPath,
    service: ServiceDep,
    player_id: Annotated[
        str | None, Query(description="Restrict to one player (statistics-service player UUID).")
    ] = None,
    stat_type: Annotated[
        str | None,
        Query(description="Restrict to one canonical stat key (e.g. 'player_points', 'player_shots')."),
    ] = None,
) -> Envelope[PlayerDistributionsData]:
    """Get per-player stat distributions for a simulation run (Phase 7 Wave 3).

    Available only for runs created with ``config.include_player_props``; 404
    when props were not captured or the run is no longer the game's latest
    (player distributions are stored latest-wins per game, like
    distributions and correlations). Stat keys are the canonical Odds API
    market keys (ADR-029); YES_NO stats carry ``yes_probability`` instead of
    an over-probability line grid.
    """
    return envelope(await service.get_player_distributions(simulation_id, player_id, stat_type))


@router.get("/simulations/{simulation_id}/correlations", response_model=Envelope[CorrelationsData])
async def get_simulation_correlations(
    simulation_id: SimulationIdPath,
    service: ServiceDep,
    legs: Annotated[
        str | None,
        Query(
            description=(
                "Comma-separated canonical leg keys (e.g. "
                "'MONEYLINE:HOME,TOTAL:OVER:2.5'). When given, the response is "
                "restricted to those legs and includes their empirical joint "
                "probability. Omit for the full default artifact."
            ),
        ),
    ] = None,
) -> Envelope[CorrelationsData]:
    """Get the same-game parlay correlation artifact for a simulation run.

    Returns leg marginals and the pairwise phi correlation matrix over the
    canonical leg vocabulary; with ?legs= the empirical joint probability of
    the requested leg set (the exact Monte Carlo joint, not a copula
    approximation). Poisson-grid sports (soccer/hockey) also expose the
    analytic joint goal grid.
    """
    leg_list = [part.strip() for part in legs.split(",") if part.strip()] if legs is not None else None
    return envelope(await service.get_correlations(simulation_id, leg_list or None))
