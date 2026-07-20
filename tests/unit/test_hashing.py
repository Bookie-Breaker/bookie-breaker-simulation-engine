"""Parameter hash stability and sensitivity tests."""

from simulation_engine.core.hashing import (
    HASH_LENGTH,
    ROSTER_SIGNATURE_LENGTH,
    compute_parameters_hash,
    compute_roster_signature,
)
from simulation_engine.core.params import GameContext, LiveState, PlayerRates

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


class TestLiveStateHashing:
    """Phase 7 Wave 2: live_state enters the hash; pregame hashes are untouched.

    live_state=None must preserve the pre-Wave-2 pinned digest byte for byte
    (the recursive None-strip removes the field entirely), while any set live
    state — and every DISTINCT live state — produces its own digest so each
    in-game snapshot gets its own cache entry.
    """

    def test_absent_live_state_preserves_pinned_pregame_digest(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        implicit = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG))
        explicit_none = compute_parameters_hash("g1", h, a, GameContext(live_state=None), dict(CONFIG))
        assert implicit == TestPluginLabel.PRE_REFACTOR_NBA_HASH
        assert explicit_none == TestPluginLabel.PRE_REFACTOR_NBA_HASH

    def test_live_state_changes_digest(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        base = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG))
        live = compute_parameters_hash(
            "g1",
            h,
            a,
            GameContext(live_state=LiveState(home_score=1, away_score=0, fraction_remaining=0.35)),
            dict(CONFIG),
        )
        assert base != live

    def test_distinct_live_states_get_distinct_digests(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")

        def digest(state: LiveState) -> str:
            return compute_parameters_hash("g1", h, a, GameContext(live_state=state), dict(CONFIG))

        states = [
            LiveState(home_score=1, away_score=0, fraction_remaining=0.35),
            LiveState(home_score=0, away_score=1, fraction_remaining=0.35),
            LiveState(home_score=1, away_score=0, fraction_remaining=0.34),
            LiveState(home_score=1, away_score=0, fraction_remaining=0.35, period=2),
            LiveState(home_score=1, away_score=0, fraction_remaining=0.35, period=2, clock_seconds=1830),
        ]
        digests = [digest(state) for state in states]
        assert len(set(digests)) == len(states)

    def test_identical_live_states_hash_identically(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        state = LiveState(home_score=2, away_score=1, fraction_remaining=0.5, period=3)
        first = compute_parameters_hash("g1", h, a, GameContext(live_state=state), dict(CONFIG))
        second = compute_parameters_hash("g1", h, a, GameContext(live_state=state), dict(CONFIG))
        assert first == second

    def test_none_fields_inside_live_state_are_stripped(self, make_team_params) -> None:
        """Unset optional refinements canonicalize away: explicit Nones == omitted."""
        h, a = make_team_params("h"), make_team_params("a")
        omitted = LiveState(home_score=1, away_score=0, fraction_remaining=0.35)
        explicit = LiveState(
            home_score=1,
            away_score=0,
            fraction_remaining=0.35,
            period=None,
            clock_seconds=None,
            bases=None,
            outs=None,
            half=None,
            possession=None,
            down=None,
            yardline=None,
        )
        a1 = compute_parameters_hash("g1", h, a, GameContext(live_state=omitted), dict(CONFIG))
        a2 = compute_parameters_hash("g1", h, a, GameContext(live_state=explicit), dict(CONFIG))
        assert a1 == a2


def make_player(player_id: str, team: str = "HOME", goal_share: float = 0.5) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name=player_id.upper(),
        position="F",
        team=team,  # type: ignore[arg-type]
        rates={"goal_share": goal_share, "shots_per_match": 2.0},
    )


class TestRosterSignatureHashing:
    """Phase 7 Wave 3: roster_signature enters the hash; pregame hashes untouched.

    roster_signature=None (props off) must preserve the pinned pregame digest
    byte for byte; a set signature — plus the PROP_ENGINE_VERSION fold that
    accompanies it — produces a distinct digest, and every distinct roster
    produces its own digest so roster changes invalidate cached props runs.
    """

    def test_absent_roster_signature_preserves_pinned_pregame_digest(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        implicit = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG))
        explicit_none = compute_parameters_hash("g1", h, a, GameContext(roster_signature=None), dict(CONFIG))
        assert implicit == TestPluginLabel.PRE_REFACTOR_NBA_HASH
        assert explicit_none == TestPluginLabel.PRE_REFACTOR_NBA_HASH

    def test_props_on_changes_digest(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")
        base = compute_parameters_hash("g1", h, a, GameContext(), dict(CONFIG))
        signature = compute_roster_signature([make_player("p1")], [make_player("p2", team="AWAY")])
        props_on = compute_parameters_hash("g1", h, a, GameContext(roster_signature=signature), dict(CONFIG))
        assert base != props_on

    def test_roster_change_changes_digest(self, make_team_params) -> None:
        h, a = make_team_params("h"), make_team_params("a")

        def digest(home: list[PlayerRates], away: list[PlayerRates]) -> str:
            signature = compute_roster_signature(home, away)
            return compute_parameters_hash("g1", h, a, GameContext(roster_signature=signature), dict(CONFIG))

        base = digest([make_player("p1"), make_player("p2")], [make_player("p3", team="AWAY")])
        dropped = digest([make_player("p1")], [make_player("p3", team="AWAY")])
        rate_drift = digest([make_player("p1"), make_player("p2", goal_share=0.6)], [make_player("p3", team="AWAY")])
        empty = digest([], [])
        assert len({base, dropped, rate_drift, empty}) == 4

    def test_signature_is_order_insensitive_and_stable(self) -> None:
        p1, p2 = make_player("p1", goal_share=0.7), make_player("p2", goal_share=0.3)
        away = [make_player("p3", team="AWAY")]
        assert compute_roster_signature([p1, p2], away) == compute_roster_signature([p2, p1], away)
        signature = compute_roster_signature([p1, p2], away)
        assert len(signature) == ROSTER_SIGNATURE_LENGTH
        assert all(c in "0123456789abcdef" for c in signature)

    def test_signature_sides_are_not_interchangeable(self) -> None:
        home = [make_player("p1")]
        away = [make_player("p2", team="AWAY")]
        assert compute_roster_signature(home, away) != compute_roster_signature(away, home)
