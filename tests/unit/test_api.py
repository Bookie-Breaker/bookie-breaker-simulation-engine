"""API layer tests with a stubbed simulation service (no Redis, no upstream)."""

from typing import Any

from fastapi.testclient import TestClient

from simulation_engine.api.dependencies import get_simulation_service
from simulation_engine.api.errors import NotFoundError
from simulation_engine.api.models import (
    LiveStateIn,
    Percentiles,
    SimulationConfigIn,
    SimulationConfigOut,
    SimulationResultData,
    SimulationRunData,
)
from simulation_engine.config import Settings
from simulation_engine.main import create_app
from simulation_engine.services.simulation_service import SimulationService


def make_run(game_id: str = "g-1", cached: bool = False) -> SimulationRunData:
    return SimulationRunData(
        simulation_run_id="run-1",
        game_id=game_id,
        status="completed",
        cached=cached,
        config=SimulationConfigOut(sport="BASKETBALL", iterations=2000, convergence_threshold=0.005),
        started_at="2026-07-04T12:00:00Z",
        completed_at="2026-07-04T12:00:01Z",
        duration_ms=55,
        iterations_completed=2000,
        converged=True,
        parameters_hash="a1b2c3d4e5f6",
        result=SimulationResultData(
            id="res-1",
            home_win_probability=0.6,
            away_win_probability=0.4,
            draw_probability=0.0,
            mean_home_score=112.4,
            mean_away_score=109.8,
            mean_total=222.2,
            mean_margin=2.6,
            spread_cover_probabilities={"-2.5": 0.52},
            total_over_probabilities={"222.5": 0.48},
            percentiles=Percentiles(margin={"50": 3}, total={"50": 222}),
        ),
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
    ) -> SimulationRunData:
        self.last_call = {
            "game_id": game_id,
            "iterations": config.iterations,
            "force_refresh": force_refresh,
            "idempotency_key": idempotency_key,
            "live_state": live_state,
        }
        return make_run(game_id)

    async def get_run(self, simulation_id: str) -> SimulationRunData:
        if simulation_id != "run-1":
            raise NotFoundError(f"Simulation run {simulation_id} not found")
        return make_run()


def make_client(stub: StubService) -> TestClient:
    app = create_app(Settings(redis_url="redis://localhost:1", statistics_service_url="http://stats.invalid"))
    app.dependency_overrides[get_simulation_service] = lambda: stub
    return TestClient(app)


class TestEnvelope:
    def test_success_envelope_shape(self) -> None:
        client = make_client(StubService())
        with client:
            response = client.post("/api/v1/sim/simulations", json={"game_id": "g-1"})
        assert response.status_code == 201
        body = response.json()
        assert body["data"]["simulation_run_id"] == "run-1"
        assert "timestamp" in body["meta"]
        assert "request_id" in body["meta"]
        assert response.headers["X-Request-ID"] == body["meta"]["request_id"]

    def test_error_envelope_shape(self) -> None:
        client = make_client(StubService())
        with client:
            response = client.get("/api/v1/sim/simulations/nope")
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "RESOURCE_NOT_FOUND"
        assert "meta" in body


class TestValidation:
    def test_iterations_above_max_rejected(self) -> None:
        client = make_client(StubService())
        with client:
            response = client.post(
                "/api/v1/sim/simulations", json={"game_id": "g-1", "config": {"iterations": 100_000}}
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_missing_game_id_rejected(self) -> None:
        client = make_client(StubService())
        with client:
            response = client.post("/api/v1/sim/simulations", json={})
        assert response.status_code == 400

    def test_idempotency_key_forwarded(self) -> None:
        stub = StubService()
        client = make_client(stub)
        with client:
            client.post(
                "/api/v1/sim/simulations",
                json={"game_id": "g-1"},
                headers={"X-Idempotency-Key": "idem-1"},
            )
        assert stub.last_call["idempotency_key"] == "idem-1"

    def test_defaults_applied(self) -> None:
        stub = StubService()
        client = make_client(stub)
        with client:
            client.post("/api/v1/sim/simulations", json={"game_id": "g-1"})
        assert stub.last_call["iterations"] == 10_000
        assert stub.last_call["force_refresh"] is False
        assert stub.last_call["live_state"] is None


LIVE_STATE = {
    "home_score": 1,
    "away_score": 0,
    "fraction_remaining": 0.35,
    "period": 2,
    "clock_seconds": 1830,
}


class _SentinelStatistics:
    """Marks the moment validation is passed: get_game raises a clean 404."""

    async def get_game(self, game_id: str):
        raise NotFoundError(f"validation passed; game {game_id} lookup reached the statistics client")


def make_real_service_client() -> TestClient:
    """Client wired to a REAL service with sentinel dependencies.

    Live-state validation and the batch live_state rejection both run before
    the service touches its cache, statistics client, or Redis, so 422 paths
    are exercisable without any backing infrastructure; a request that passes
    validation surfaces the sentinel's 404 instead.
    """
    settings = Settings(redis_url="redis://localhost:1", statistics_service_url="http://stats.invalid")
    service = SimulationService(settings, cache=None, statistics=_SentinelStatistics(), redis_client=None)  # type: ignore[arg-type]
    app = create_app(settings)
    app.dependency_overrides[get_simulation_service] = lambda: service
    return TestClient(app)


class TestLiveState:
    """Phase 7 Wave 2: live_state plumbing and 422 validation contract."""

    def test_live_state_forwarded_to_service(self) -> None:
        stub = StubService()
        client = make_client(stub)
        with client:
            response = client.post("/api/v1/sim/simulations", json={"game_id": "g-1", "live_state": LIVE_STATE})
        assert response.status_code == 201
        forwarded = stub.last_call["live_state"]
        assert isinstance(forwarded, LiveStateIn)
        assert forwarded.home_score == 1
        assert forwarded.fraction_remaining == 0.35
        assert forwarded.bases is None

    def post_live(self, client: TestClient, **overrides: object):
        return client.post(
            "/api/v1/sim/simulations",
            json={"game_id": "g-1", "live_state": {**LIVE_STATE, **overrides}},
        )

    def test_out_of_bounds_live_state_rejected_with_422(self) -> None:
        client = make_real_service_client()
        with client:
            for overrides in (
                {"fraction_remaining": 0.0},
                {"fraction_remaining": 1.01},
                {"home_score": -1},
                {"away_score": -1},
                {"period": 0},
                {"clock_seconds": -1},
                {"bases": "XYZ"},
                {"outs": 3},
                {"half": "MIDDLE"},
                {"possession": "NEITHER"},
                {"down": 5},
                {"yardline": 101},
            ):
                response = self.post_live(client, **overrides)
                assert response.status_code == 422, (overrides, response.text)
                assert response.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"

    def test_boundary_values_pass_validation(self) -> None:
        """fraction_remaining=1.0 and zeroed scores are valid: the request
        proceeds past validation to the sentinel statistics client (404)."""
        client = make_real_service_client()
        with client:
            response = self.post_live(client, fraction_remaining=1.0, home_score=0, away_score=0)
        assert response.status_code == 404

    def test_batch_with_live_state_rejected_with_422(self) -> None:
        client = make_real_service_client()
        with client:
            response = client.post(
                "/api/v1/sim/simulations/batch",
                json={"games": [{"game_id": "g-1", "live_state": LIVE_STATE}, {"game_id": "g-2"}]},
            )
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "UNPROCESSABLE_ENTITY"
        assert "batch" in body["error"]["message"]
        assert "g-1" in body["error"]["message"]
