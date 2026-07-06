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


class TestPluginLabel:
    # Hex digest produced by the pre-Phase-6 implementation (hardcoded
    # "basketball" engine label) for the conftest default parameters,
    # computed on main at commit 1d49f09 before the plugin_label refactor.
    PRE_REFACTOR_NBA_HASH = "713ff2a6c08b"

    def test_nba_parity_with_pre_refactor_hash(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        assert compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG)) == self.PRE_REFACTOR_NBA_HASH
        explicit = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG), plugin_label="basketball")
        assert explicit == self.PRE_REFACTOR_NBA_HASH

    def test_sensitive_to_plugin_label(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        base = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG))
        assert base != compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG), plugin_label="soccer")


class TestContextNoneStripping:
    """Phase 6 Wave 2: None-valued context fields are stripped from the hash.

    GameContext gained optional probable-starter fields; unset (None) fields
    must not change hashes computed before the fields existed, while a set
    field must (a starter announcement invalidates cached simulations).
    """

    def test_none_starters_preserve_pre_change_hash(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        implicit = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG))
        explicit_none = compute_parameters_hash(
            "g1", h, a, GameContext(home_starter_fip=None, away_starter_fip=None), dict(CONFIG)
        )
        assert implicit == TestPluginLabel.PRE_REFACTOR_NBA_HASH
        assert explicit_none == TestPluginLabel.PRE_REFACTOR_NBA_HASH

    def test_set_starter_changes_hash(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        base = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG))
        home_set = compute_parameters_hash("g1", h, a, GameContext(home_starter_fip=3.5), dict(CONFIG))
        away_set = compute_parameters_hash("g1", h, a, GameContext(away_starter_fip=3.5), dict(CONFIG))
        both_set = compute_parameters_hash(
            "g1", h, a, GameContext(home_starter_fip=3.5, away_starter_fip=3.5), dict(CONFIG)
        )
        assert len({base, home_set, away_set, both_set}) == 4
