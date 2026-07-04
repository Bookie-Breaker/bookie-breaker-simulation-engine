"""Simulation endpoints per api-contracts/simulation-engine-api.md."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Query

from simulation_engine.api.dependencies import get_simulation_service
from simulation_engine.api.envelope import Envelope, envelope
from simulation_engine.api.models import (
    BatchData,
    BatchRequest,
    DistributionsData,
    DistributionType,
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
    force_refresh is false.
    """
    run = await service.run_simulation(
        request.game_id,
        request.config,
        force_refresh=request.force_refresh,
        idempotency_key=x_idempotency_key,
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
