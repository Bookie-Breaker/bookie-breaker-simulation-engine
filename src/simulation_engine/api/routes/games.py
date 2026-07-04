"""Game-scoped simulation lookups."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from simulation_engine.api.dependencies import get_simulation_service
from simulation_engine.api.envelope import Envelope, envelope
from simulation_engine.api.models import SimulationRunData
from simulation_engine.services.simulation_service import SimulationService

router = APIRouter(tags=["games"])


@router.get("/games/{game_id}/latest", response_model=Envelope[SimulationRunData])
async def get_latest_simulation(
    game_id: Annotated[str, Path(description="The statistics-service game identifier.")],
    service: Annotated[SimulationService, Depends(get_simulation_service)],
    force_refresh: Annotated[
        bool, Query(description="If true, trigger a fresh default-config simulation instead of returning cached.")
    ] = False,
    config_id: Annotated[
        str | None,
        Query(description="Reserved for stored simulation configs; accepted and ignored in Phase 2."),
    ] = None,
) -> Envelope[SimulationRunData]:
    """Get the most recent simulation results for a game."""
    _ = config_id
    return envelope(await service.get_latest(game_id, force_refresh=force_refresh))
