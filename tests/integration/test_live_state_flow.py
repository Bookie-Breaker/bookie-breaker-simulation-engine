"""Live re-simulation flows against real Redis with mocked statistics-service.

Phase 7 Wave 2: POST /simulations with ``live_state`` runs the remainder of
the game. Verifies the pinned contract end to end: 201 with the standard
SimulationRunData envelope, a parameters_hash DISTINCT from the pregame run
of the same game (live runs never collide with pregame cache entries),
retrievable distributions for the live run, distinct-live-state cache
entries, and the 422 rejections (bounds, batch).
"""

import uuid

BASE_CONFIG = {"iterations": 2000, "random_seed": 42}
LIVE_STATE = {
    "home_score": 55,
    "away_score": 48,
    "fraction_remaining": 0.35,
    "period": 3,
    "clock_seconds": 480,
}


def post_simulation(client, game_id: str, **overrides):
    return client.post("/api/v1/sim/simulations", json={"game_id": game_id, "config": BASE_CONFIG, **overrides})


class TestLiveSimulationFlow:
    def test_live_run_gets_distinct_hash_and_retrievable_distributions(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id, status="LIVE")

        pregame = post_simulation(client, game_id)
        assert pregame.status_code == 201, pregame.text
        pregame_data = pregame.json()["data"]

        live = post_simulation(client, game_id, live_state=LIVE_STATE)
        assert live.status_code == 201, live.text
        live_data = live.json()["data"]

        # The live run is a fresh, completed run with its own cache identity.
        assert live_data["status"] == "completed"
        assert live_data["cached"] is False
        assert live_data["simulation_run_id"] != pregame_data["simulation_run_id"]
        assert live_data["parameters_hash"] != pregame_data["parameters_hash"]

        # Remainder-of-game semantics: scores never fall below the current
        # score, so the mean total sits at or above the live sum.
        assert live_data["result"]["mean_total"] >= 55 + 48
        assert live_data["result"]["mean_home_score"] >= 55
        assert live_data["result"]["mean_away_score"] >= 48

        # Distributions for the live run are retrievable...
        run_id = live_data["simulation_run_id"]
        distributions = client.get(f"/api/v1/sim/simulations/{run_id}/distributions")
        assert distributions.status_code == 200
        dist_data = distributions.json()["data"]
        assert set(dist_data["distributions"]) == {"home_score", "away_score", "margin", "total"}
        assert dist_data["distributions"]["total"]["min"] >= 55 + 48

        # ...and the game-scoped blobs are latest-wins by design: the pregame
        # run's distributions are superseded (404), never silently wrong.
        superseded = client.get(f"/api/v1/sim/simulations/{pregame_data['simulation_run_id']}/distributions")
        assert superseded.status_code == 404

    def test_correlations_follow_latest_wins_semantics(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id, status="LIVE")

        pregame_data = post_simulation(client, game_id).json()["data"]
        live_data = post_simulation(client, game_id, live_state=LIVE_STATE).json()["data"]

        live_correlations = client.get(f"/api/v1/sim/simulations/{live_data['simulation_run_id']}/correlations")
        assert live_correlations.status_code == 200
        payload = live_correlations.json()["data"]
        assert payload["legs"]
        assert payload["simulation_run_id"] == live_data["simulation_run_id"]

        superseded = client.get(f"/api/v1/sim/simulations/{pregame_data['simulation_run_id']}/correlations")
        assert superseded.status_code == 404

    def test_identical_live_state_replays_from_cache_distinct_state_reruns(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id, status="LIVE")

        first = post_simulation(client, game_id, live_state=LIVE_STATE).json()["data"]
        replay = post_simulation(client, game_id, live_state=LIVE_STATE).json()["data"]
        assert replay["cached"] is True
        assert replay["simulation_run_id"] == first["simulation_run_id"]

        # A different in-game snapshot is a different cache entry.
        moved_on = post_simulation(
            client, game_id, live_state={**LIVE_STATE, "home_score": 60, "fraction_remaining": 0.25}
        ).json()["data"]
        assert moved_on["cached"] is False
        assert moved_on["parameters_hash"] != first["parameters_hash"]

    def test_out_of_bounds_live_state_is_422(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id, status="LIVE")
        response = post_simulation(client, game_id, live_state={**LIVE_STATE, "fraction_remaining": 0.0})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"

    def test_batch_rejects_live_state_with_422(self, client, mock_game) -> None:
        game_id = f"game-{uuid.uuid4()}"
        mock_game(game_id, status="LIVE")
        response = client.post(
            "/api/v1/sim/simulations/batch",
            json={
                "games": [{"game_id": game_id, "live_state": LIVE_STATE}],
                "default_config": BASE_CONFIG,
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"
