"""Health endpoint with dependency and load status."""

from typing import Annotated

from fastapi import APIRouter, Depends

from simulation_engine.api.dependencies import get_simulation_service
from simulation_engine.api.envelope import Envelope, envelope
from simulation_engine.api.models import HealthData
from simulation_engine.services.simulation_service import SimulationService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Envelope[HealthData])
async def get_health(
    service: Annotated[SimulationService, Depends(get_simulation_service)],
) -> Envelope[HealthData]:
    """Liveness plus dependency status. Returns 200 even when degraded so
    container healthchecks measure this service, not its dependencies."""
    return envelope(await service.health())
