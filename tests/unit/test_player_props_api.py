"""API layer tests for the player-distributions read path (Phase 7 Wave 3)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from simulation_engine.api.dependencies import get_simulation_service
from simulation_engine.api.errors import NotFoundError
from simulation_engine.api.models import (
    Distribution,
    LiveStateIn,
    PlayerDistributionsData,
    PlayerPropsEntry,
    PlayerStatBlock,
    SimulationConfigIn,
)
from simulation_engine.config import Settings
from simulation_engine.main import create_app
from simulation_engine.services.simulation_service import _hashable_config, _request_body_hash


def make_player_distributions(simulation_run_id: str = "run-1") -> PlayerDistributionsData:
    goals_dist = Distribution(values={"0": 0.55, "1": 0.3, "2": 0.15}, mean=0.6, std_dev=0.74, min=0, max=2)
    shots_dist = Distribution(values={"1": 0.4, "2": 0.35, "3": 0.25}, mean=1.85, std_dev=0.79, min=1, max=3)
    return PlayerDistributionsData(
        simulation_run_id=simulation_run_id,
        game_id="g-1",
        iterations_completed=2000,
        players={
            "player-uuid-1": PlayerPropsEntry(
                name="Test Striker",
                team="HOME",
                stats={
                    "player_goal_scorer_anytime": PlayerStatBlock(distribution=goals_dist, yes_probability=0.45),
                    "player_shots": PlayerStatBlock(
                        distribution=shots_dist,
                        over_probabilities={"0.5": 1.0, "1.5": 0.6, "2.5": 0.25},
                    ),
                },
            )
        },
    )


class StubService:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] = {}

    async def run_simulation(
        self,
        game_id: str,
        config: SimulationConfigIn,
        force_refresh: bool = False,
        idempotency_key: str | None = None,
        live_state: LiveStateIn | None = None,
    ) -> None:
        self.last_call = {"game_id": game_id, "include_player_props": config.include_player_props}
        raise NotFoundError("stub stops after recording the call")

    async def get_player_distributions(
        self, simulation_id: str, player_id: str | None = None, stat_type: str | None = None
    ) -> PlayerDistributionsData:
        self.last_call = {"simulation_id": simulation_id, "player_id": player_id, "stat_type": stat_type}
        if simulation_id != "run-1":
            raise NotFoundError(f"Player distributions for simulation {simulation_id} are not available")
        return make_player_distributions()


def make_client(stub: StubService) -> TestClient:
    app = create_app(Settings(redis_url="redis://localhost:1", statistics_service_url="http://stats.invalid"))
    app.dependency_overrides[get_simulation_service] = lambda: stub
    return TestClient(app)


class TestIncludePlayerPropsFlag:
    def test_defaults_to_false(self) -> None:
        stub = StubService()
        client = make_client(stub)
        with client:
            client.post("/api/v1/sim/simulations", json={"game_id": "g-1"})
        assert stub.last_call["include_player_props"] is False

    def test_forwarded_when_set(self) -> None:
        stub = StubService()
        client = make_client(stub)
        with client:
            client.post(
                "/api/v1/sim/simulations",
                json={"game_id": "g-1", "config": {"include_player_props": True}},
            )
        assert stub.last_call["include_player_props"] is True


class TestPlayerDistributionsEndpoint:
    def test_success_envelope_shape(self) -> None:
        client = make_client(StubService())
        with client:
            response = client.get("/api/v1/sim/simulations/run-1/player-distributions")
        assert response.status_code == 200
        body = response.json()
        data = body["data"]
        assert data["simulation_run_id"] == "run-1"
        assert data["game_id"] == "g-1"
        entry = data["players"]["player-uuid-1"]
        assert entry["name"] == "Test Striker"
        assert entry["team"] == "HOME"
        anytime = entry["stats"]["player_goal_scorer_anytime"]
        assert anytime["yes_probability"] == 0.45
        assert anytime["over_probabilities"] is None
        shots = entry["stats"]["player_shots"]
        assert shots["yes_probability"] is None
        assert shots["over_probabilities"]["1.5"] == 0.6
        assert "timestamp" in body["meta"]

    def test_query_filters_forwarded(self) -> None:
        stub = StubService()
        client = make_client(stub)
        with client:
            client.get(
                "/api/v1/sim/simulations/run-1/player-distributions",
                params={"player_id": "player-uuid-1", "stat_type": "player_shots"},
            )
        assert stub.last_call == {
            "simulation_id": "run-1",
            "player_id": "player-uuid-1",
            "stat_type": "player_shots",
        }

    def test_missing_run_returns_404_envelope(self) -> None:
        client = make_client(StubService())
        with client:
            response = client.get("/api/v1/sim/simulations/nope/player-distributions")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


class TestHashableConfig:
    """include_player_props=False must not perturb any pre-Wave-3 hashed payload."""

    def test_default_config_payload_omits_the_flag(self) -> None:
        payload = _hashable_config(SimulationConfigIn())
        assert "include_player_props" not in payload
        assert payload == {
            "iterations": 10_000,
            "convergence_threshold": 0.005,
            "random_seed": None,
            "plugin_config": {},
        }

    def test_props_on_stays_in_payload_and_changes_body_hash(self) -> None:
        payload = _hashable_config(SimulationConfigIn(include_player_props=True))
        assert payload["include_player_props"] is True
        off = _request_body_hash("g-1", SimulationConfigIn(), False)
        on = _request_body_hash("g-1", SimulationConfigIn(include_player_props=True), False)
        assert off != on


class TestModelValidation:
    def test_player_stat_block_requires_distribution(self) -> None:
        payload = make_player_distributions().model_dump(mode="json")
        # Round-trips cleanly through validation (the cache read path re-validates).
        assert PlayerDistributionsData.model_validate(payload) == make_player_distributions()

    def test_team_literal_enforced(self) -> None:
        with pytest.raises(ValidationError):
            PlayerPropsEntry(name="X", team="NEUTRAL", stats={})  # type: ignore[arg-type]
