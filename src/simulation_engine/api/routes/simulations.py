"""Simulation endpoints per api-contracts/simulation-engine-api.md."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query

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


@router.post("/simulations", status_code=201, response_model=Envelope[SimulationRunData])
async def create_simulation(
    request: SimulationRequest,
    service: ServiceDep,
    x_idempotency_key: Annotated[str | None, Header()] = None,
) -> Envelope[SimulationRunData]:
    run = await service.run_simulation(
        request.game_id,
        request.config,
        force_refresh=request.force_refresh,
        idempotency_key=x_idempotency_key,
    )
    return envelope(run)


@router.post("/simulations/batch", status_code=201, response_model=Envelope[BatchData])
async def create_batch(request: BatchRequest, service: ServiceDep) -> Envelope[BatchData]:
    batch = await service.run_batch(request.games, request.default_config, request.force_refresh)
    return envelope(batch)


@router.get("/simulations/{simulation_id}", response_model=Envelope[SimulationRunData])
async def get_simulation(simulation_id: str, service: ServiceDep) -> Envelope[SimulationRunData]:
    return envelope(await service.get_run(simulation_id))


@router.get("/simulations/{simulation_id}/distributions", response_model=Envelope[DistributionsData])
async def get_simulation_distributions(
    simulation_id: str,
    service: ServiceDep,
    distribution_type: Annotated[DistributionType, Query()] = "all",
) -> Envelope[DistributionsData]:
    return envelope(await service.get_distributions(simulation_id, distribution_type))
