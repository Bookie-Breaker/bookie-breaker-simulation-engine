"""End-to-end simulation flows against real Redis with mocked statistics-service."""

import json
import time
import uuid

import redis as sync_redis
from httpx import Response

BODY = {"game_id": "", "config": {"iterations": 2000, "random_seed": 42}}


def post_simulation(client, game_id: str, **overrides):
    body = {**BODY, "game_id": game_id, **overrides}
    return client.post("/api/v1/sim/simulations", json=body)


class TestSimulateAndPersist:
    def test_simulation_persists_and_publishes(self, client, mock_game, redis_url) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id)

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe("events:simulation.completed")

        response = post_simulation(client, game_id)
        assert response.status_code == 201, response.text
        data = response.json()["data"]
        assert data["status"] == "completed"
        assert data["cached"] is False
        assert data["iterations_completed"] <= 2000
        assert data["config"]["sport"] == "BASKETBALL"
        assert 0.0 < data["result"]["home_win_probability"] < 1.0
        assert data["result"]["spread_cover_probabilities"]

        # Redis keys exist with TTLs (redis-schemas.md layout)
        result_key = f"sim:result:{game_id}:{data['parameters_hash']}"
        assert redis_client.hget(result_key, "simulation_run_id") == data["simulation_run_id"]
        assert 0 < redis_client.ttl(result_key) <= 7200
        assert redis_client.ttl(f"sim:run:{data['simulation_run_id']}") > 0
        assert redis_client.get(f"sim:latest:{game_id}") == data["simulation_run_id"]

        # simulation.completed published with the documented payload;
        # poll: get_message returns None when it swallows the subscribe ack
        message = None
        deadline = time.monotonic() + 5.0
        while message is None and time.monotonic() < deadline:
            message = pubsub.get_message(timeout=0.5)
        assert message is not None, "expected a simulation.completed event"
        event = json.loads(message["data"])
        assert event["event"] == "simulation.completed"
        assert event["simulation_run_id"] == data["simulation_run_id"]
        assert event["league"] == "NBA"
        pubsub.close()

    def test_second_identical_request_is_cached(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id)

        first = post_simulation(client, game_id).json()["data"]
        second = post_simulation(client, game_id).json()["data"]
        assert second["cached"] is True
        assert second["simulation_run_id"] == first["simulation_run_id"]

        refreshed = post_simulation(client, game_id, force_refresh=True).json()["data"]
        assert refreshed["cached"] is False
        assert refreshed["simulation_run_id"] != first["simulation_run_id"]

    def test_completed_game_rejected(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id, status="FINAL")
        response = post_simulation(client, game_id)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"


class TestIdempotency:
    def test_same_key_same_body_replays(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id)
        key = str(uuid.uuid4())

        first = client.post(
            "/api/v1/sim/simulations",
            json={**BODY, "game_id": game_id},
            headers={"X-Idempotency-Key": key},
        )
        replay = client.post(
            "/api/v1/sim/simulations",
            json={**BODY, "game_id": game_id},
            headers={"X-Idempotency-Key": key},
        )
        assert replay.status_code == 201
        assert replay.json()["data"]["simulation_run_id"] == first.json()["data"]["simulation_run_id"]

    def test_same_key_different_body_conflicts(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id)
        key = str(uuid.uuid4())

        client.post("/api/v1/sim/simulations", json={**BODY, "game_id": game_id}, headers={"X-Idempotency-Key": key})
        conflict = client.post(
            "/api/v1/sim/simulations",
            json={"game_id": game_id, "config": {"iterations": 1900, "random_seed": 7}},
            headers={"X-Idempotency-Key": key},
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "DUPLICATE_RESOURCE"


class TestLookups:
    def test_run_latest_and_distributions_round_trip(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id)
        created = post_simulation(client, game_id).json()["data"]
        run_id = created["simulation_run_id"]

        by_id = client.get(f"/api/v1/sim/simulations/{run_id}")
        assert by_id.status_code == 200
        assert by_id.json()["data"]["game_id"] == game_id

        latest = client.get(f"/api/v1/sim/games/{game_id}/latest")
        assert latest.status_code == 200
        assert latest.json()["data"]["simulation_run_id"] == run_id

        distributions = client.get(f"/api/v1/sim/simulations/{run_id}/distributions")
        assert distributions.status_code == 200
        dist_data = distributions.json()["data"]
        assert set(dist_data["distributions"]) == {"home_score", "away_score", "margin", "total"}

        margin_only = client.get(f"/api/v1/sim/simulations/{run_id}/distributions?distribution_type=margin")
        assert set(margin_only.json()["data"]["distributions"]) == {"margin"}

    def test_unknown_run_and_game_return_404(self, client) -> None:
        assert client.get(f"/api/v1/sim/simulations/{uuid.uuid4()}").status_code == 404
        assert client.get(f"/api/v1/sim/games/{uuid.uuid4()}/latest").status_code == 404


class TestBatchAndHealth:
    def test_batch_with_one_failure_is_partial(self, client, mock_game, stats_service) -> None:
        good_game = f"game-{uuid.uuid4()}"
        missing_game = f"game-{uuid.uuid4()}"
        mock_game(good_game)
        stats_service.get(f"/api/v1/stats/games/{missing_game}").mock(
            return_value=Response(
                404,
                json={"error": {"code": "NOT_FOUND", "message": "no such game"}, "meta": {}},
            )
        )

        response = client.post(
            "/api/v1/sim/simulations/batch",
            json={
                "games": [{"game_id": good_game}, {"game_id": missing_game}],
                "default_config": {"iterations": 2000, "random_seed": 1},
            },
        )
        assert response.status_code == 201
        data = response.json()["data"]
        assert data["status"] == "partial"
        assert data["completed_games"] == 1
        assert data["failed_games"] == 1
        statuses = {r["game_id"]: r["status"] for r in data["results"]}
        assert statuses[good_game] == "completed"
        assert statuses[missing_game] == "failed"

    def test_health_reports_dependencies(self, client, stats_service, payloads) -> None:
        stats_service.get("/api/v1/stats/health").mock(
            return_value=Response(200, json=payloads["envelope"]({"status": "healthy"}))
        )
        response = client.get("/api/v1/sim/health")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["service"] == "simulation-engine"
        assert data["dependencies"]["redis"] == "healthy"
        assert data["dependencies"]["statistics_service"] == "healthy"
        assert data["load"]["max_concurrent"] == 30

    def test_non_envelope_upstream_is_dependency_error(self, client, stats_service, payloads) -> None:
        game_id = f"game-{uuid.uuid4()}"
        stats_service.get(f"/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=payloads["game"](game_id))
        )
        response = post_simulation(client, game_id)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "DEPENDENCY_ERROR"
