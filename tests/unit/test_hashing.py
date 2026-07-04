"""Parameter hash stability and sensitivity tests."""

from simulation_engine.core.hashing import HASH_LENGTH, compute_parameters_hash
from simulation_engine.core.params import GameContext

CONFIG = {"iterations": 10_000, "convergence_threshold": 0.005, "random_seed": None, "plugin_config": {}}


class TestParametersHash:
    def test_stable_for_identical_inputs(self, make_team_params) -> None:
        a = compute_parameters_hash("g1", make_team_params("h"), make_team_params("a"), GameContext(), dict(CONFIG))
        b = compute_parameters_hash("g1", make_team_params("h"), make_team_params("a"), GameContext(), dict(CONFIG))
        assert a == b
        assert len(a) == HASH_LENGTH
        assert all(c in "0123456789abcdef" for c in a)

    def test_insensitive_to_float_noise_below_rounding(self, make_team_params) -> None:
        base = compute_parameters_hash("g1", make_team_params("h"), make_team_params("a"), GameContext(), dict(CONFIG))
        noisy = compute_parameters_hash(
            "g1", make_team_params("h", pace=100.0 + 1e-9), make_team_params("a"), GameContext(), dict(CONFIG)
        )
        assert base == noisy

    def test_sensitive_to_team_params(self, make_team_params) -> None:
        base = compute_parameters_hash("g1", make_team_params("h"), make_team_params("a"), GameContext(), dict(CONFIG))
        changed = compute_parameters_hash(
            "g1", make_team_params("h", off_rating=115.0), make_team_params("a"), GameContext(), dict(CONFIG)
        )
        assert base != changed

    def test_sensitive_to_config_and_game(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        base = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG))
        assert base != compute_parameters_hash("g2", h, a, GameContext(), dict(CONFIG))
        assert base != compute_parameters_hash("g1", h, a, GameContext(), {**CONFIG, "random_seed": 42})
        assert base != compute_parameters_hash("g1", h, a, GameContext(), {**CONFIG, "iterations": 20_000})
