"""FastAPI dependency accessors backed by app.state."""

from fastapi import Request

from simulation_engine.services.simulation_service import SimulationService


def get_simulation_service(request: Request) -> SimulationService:
    service: SimulationService = request.app.state.simulation_service
    return service
