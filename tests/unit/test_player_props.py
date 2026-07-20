"""Player-prop detailed-path tests (Phase 7 Wave 3): plugins, runner, and output.

Conservation is exact by construction (multinomial allocation), so those
assertions are equality checks per iteration, not statistical tolerances.
"""

import numpy as np
import numpy.typing as npt
import pytest

from simulation_engine.core.output import build_player_distributions, default_line_grid
from simulation_engine.core.params import GameContext, PlayerRates
from simulation_engine.core.plugins import SOCCER_GRID_CONFIG, get_simulator
from simulation_engine.core.plugins.baseball import BaseballParams, BaseballSimulator
from simulation_engine.core.plugins.basketball import BasketballSimulator
from simulation_engine.core.plugins.football import FootballParams, FootballSimulator
from simulation_engine.core.plugins.soccer import SoccerParams, SoccerSimulator
from simulation_engine.core.runner import SimulationOutput, run_monte_carlo

N = 5000


def soccer_rates(player_id: str, team: str, goal_share: float, shots: float = 2.0, sot: float = 0.9) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name=player_id.upper(),
        position="F",
        team=team,  # type: ignore[arg-type]
        rates={"goal_share": goal_share, "shots_per_match": shots, "sot_per_match": sot, "minutes_share": 0.9},
    )


def nba_rates(
    player_id: str, team: str, weight: float, reb: float = 6.0, ast: float = 4.0, th3: float = 2.0
) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name=player_id.upper(),
        position="G",
        team=team,  # type: ignore[arg-type]
        rates={
            "points_weight": weight,
            "rebounds_per_game": reb,
            "assists_per_game": ast,
            "threes_per_game": th3,
            "minutes_share": 0.7,
        },
    )


def make_soccer_sim(home_roster: list[PlayerRates], away_roster: list[PlayerRates]) -> SoccerSimulator:
    sim = SoccerSimulator({})
    sim.set_players(home_roster, away_roster)
    home = SoccerParams(attack=1.15, defense=0.9, goals_for_per_match=1.6, goals_against_per_match=1.1)
    away = SoccerParams(attack=0.95, defense=1.05, goals_for_per_match=1.3, goals_against_per_match=1.4)
    sim.set_parameters(home, away, GameContext(league="FIFA_WC", neutral_site=True))
    return sim


HOME_SOCCER = [soccer_rates("h1", "HOME", 0.5), soccer_rates("h2", "HOME", 0.3), soccer_rates("h3", "HOME", 0.2)]
AWAY_SOCCER = [soccer_rates("a1", "AWAY", 0.7), soccer_rates("a2", "AWAY", 0.3)]


class TestSoccerDetailed:
    def test_goal_conservation_per_iteration(self) -> None:
        sim = make_soccer_sim(HOME_SOCCER, AWAY_SOCCER)
        batch = sim.simulate_games_detailed(np.random.default_rng(7), N)
        home_allocated = sum(batch.player_stats[p.player_id]["player_goal_scorer_anytime"] for p in HOME_SOCCER)
        away_allocated = sum(batch.player_stats[p.player_id]["player_goal_scorer_anytime"] for p in AWAY_SOCCER)
        assert np.array_equal(home_allocated, batch.home_scores)
        assert np.array_equal(away_allocated, batch.away_scores)

    def test_all_canonical_soccer_keys_present_and_aligned(self) -> None:
        sim = make_soccer_sim(HOME_SOCCER, AWAY_SOCCER)
        batch = sim.simulate_games_detailed(np.random.default_rng(3), N)
        assert set(batch.player_stats) == {"h1", "h2", "h3", "a1", "a2"}
        for stats in batch.player_stats.values():
            assert set(stats) == {"player_goal_scorer_anytime", "player_shots", "player_shots_on_target"}
            for values in stats.values():
                assert len(values) == N
                assert values.dtype == np.int32
                assert values.min() >= 0

    def test_goal_shares_drive_allocation(self) -> None:
        sim = make_soccer_sim(HOME_SOCCER, [])
        batch = sim.simulate_games_detailed(np.random.default_rng(11), 50_000)
        means = {p: float(np.mean(batch.player_stats[p]["player_goal_scorer_anytime"])) for p in ("h1", "h2", "h3")}
        team_mean = float(np.mean(batch.home_scores))
        assert means["h1"] == pytest.approx(0.5 * team_mean, rel=0.05)
        assert means["h1"] > means["h2"] > means["h3"]

    def test_shot_rates_match_poisson_mean(self) -> None:
        sim = make_soccer_sim([soccer_rates("h1", "HOME", 1.0, shots=2.5, sot=1.1)], [])
        batch = sim.simulate_games_detailed(np.random.default_rng(5), 50_000)
        assert float(np.mean(batch.player_stats["h1"]["player_shots"])) == pytest.approx(2.5, rel=0.05)
        assert float(np.mean(batch.player_stats["h1"]["player_shots_on_target"])) == pytest.approx(1.1, rel=0.05)

    def test_empty_rosters_yield_team_scores_and_no_player_output(self) -> None:
        sim = make_soccer_sim([], [])
        batch = sim.simulate_games_detailed(np.random.default_rng(1), 1000)
        assert batch.player_stats == {}
        assert len(batch.home_scores) == 1000

    def test_zero_share_roster_does_not_crash(self) -> None:
        sim = make_soccer_sim([soccer_rates("h1", "HOME", 0.0)], [])
        batch = sim.simulate_games_detailed(np.random.default_rng(1), 500)
        assert batch.player_stats == {}


class TestBasketballDetailed:
    def make_sim(
        self, home_roster: list[PlayerRates], away_roster: list[PlayerRates], make_team_params
    ) -> BasketballSimulator:
        sim = BasketballSimulator({})
        sim.set_players(home_roster, away_roster)
        sim.set_parameters(make_team_params("h"), make_team_params("a"), GameContext())
        return sim

    HOME = [nba_rates("h1", "HOME", 20.0), nba_rates("h2", "HOME", 12.0), nba_rates("h3", "HOME", 8.0)]
    AWAY = [nba_rates("a1", "AWAY", 18.0), nba_rates("a2", "AWAY", 10.0)]

    def test_points_conservation_per_iteration(self, make_team_params) -> None:
        sim = self.make_sim(self.HOME, self.AWAY, make_team_params)
        batch = sim.simulate_games_detailed(np.random.default_rng(9), 2000)
        home_pts = sum(batch.player_stats[p.player_id]["player_points"] for p in self.HOME)
        away_pts = sum(batch.player_stats[p.player_id]["player_points"] for p in self.AWAY)
        assert np.array_equal(home_pts, batch.home_scores)
        assert np.array_equal(away_pts, batch.away_scores)

    def test_pra_is_sum_of_components(self, make_team_params) -> None:
        sim = self.make_sim(self.HOME, self.AWAY, make_team_params)
        batch = sim.simulate_games_detailed(np.random.default_rng(2), 2000)
        for stats in batch.player_stats.values():
            assert np.array_equal(
                stats["player_points_rebounds_assists"],
                stats["player_points"] + stats["player_rebounds"] + stats["player_assists"],
            )

    def test_all_canonical_basketball_keys_present(self, make_team_params) -> None:
        sim = self.make_sim(self.HOME, self.AWAY, make_team_params)
        batch = sim.simulate_games_detailed(np.random.default_rng(4), 500)
        for stats in batch.player_stats.values():
            assert set(stats) == {
                "player_points",
                "player_rebounds",
                "player_assists",
                "player_threes",
                "player_points_rebounds_assists",
            }

    def test_rebound_rate_tracks_season_value(self, make_team_params) -> None:
        sim = self.make_sim([nba_rates("h1", "HOME", 20.0, reb=8.5)], [], make_team_params)
        batch = sim.simulate_games_detailed(np.random.default_rng(6), 30_000)
        # Pace factor averages ~1.0 over the possession distribution.
        assert float(np.mean(batch.player_stats["h1"]["player_rebounds"])) == pytest.approx(8.5, rel=0.05)

    def test_no_players_delegates_to_team_path(self, make_team_params) -> None:
        sim = self.make_sim([], [], make_team_params)
        batch = sim.simulate_games_detailed(np.random.default_rng(1), 500)
        assert batch.player_stats == {}
        assert len(batch.home_scores) == 500
        assert not np.any(batch.home_scores == batch.away_scores)  # OT still resolves ties


class TestDormantSports:
    def test_baseball_stores_roster_and_returns_empty_player_stats(self) -> None:
        sim = BaseballSimulator({})
        sim.set_players([soccer_rates("x", "HOME", 1.0)], [])
        params = BaseballParams(
            runs_scored_per_game=4.5,
            runs_allowed_per_game=4.3,
            team_era=4.0,
            team_fip=4.0,
            bullpen_era=4.1,
        )
        sim.set_parameters(params, params, GameContext(league="MLB"))
        batch = sim.simulate_games_detailed(np.random.default_rng(1), 200)
        assert batch.player_stats == {}
        assert len(batch.home_scores) == 200

    def test_football_stores_roster_and_returns_empty_player_stats(self) -> None:
        sim = FootballSimulator({})
        sim.set_players([], [soccer_rates("y", "AWAY", 1.0)])
        params = FootballParams(
            points_per_game=24.0,
            points_allowed_per_game=21.0,
            drives_per_game=11.0,
            points_per_drive_off=2.1,
            points_per_drive_def=1.9,
            epa_per_play_off=0.0,
            epa_per_play_def=0.0,
        )
        sim.set_parameters(params, params, GameContext(league="NFL"))
        batch = sim.simulate_games_detailed(np.random.default_rng(1), 200)
        assert batch.player_stats == {}
        assert len(batch.home_scores) == 200

    def test_hockey_default_detailed_contract(self) -> None:
        sim = get_simulator("NHL")
        sim.set_players([], [])  # base-class no-op must not raise


class TestRunnerCapture:
    def run(self, capture: bool, iterations: int = 6000, **kwargs) -> SimulationOutput:
        sim = SoccerSimulator({})
        sim.set_players(HOME_SOCCER, AWAY_SOCCER)
        return run_monte_carlo(
            sim,
            SoccerParams(attack=1.1, defense=0.9, goals_for_per_match=1.5, goals_against_per_match=1.2),
            SoccerParams(attack=1.0, defense=1.0, goals_for_per_match=1.35, goals_against_per_match=1.35),
            GameContext(league="FIFA_WC", neutral_site=True),
            iterations=iterations,
            seed=42,
            grid_config=SOCCER_GRID_CONFIG,
            capture_players=capture,
            **kwargs,
        )

    def test_capture_off_yields_no_player_stats(self) -> None:
        out = self.run(capture=False, convergence_threshold=1e-9)
        assert out.player_stats == {}

    def test_capture_concatenates_chunks_aligned_with_scores(self) -> None:
        out = self.run(capture=True, convergence_threshold=1e-9)
        assert out.iterations_run == 6000
        assert set(out.player_stats) == {"h1", "h2", "h3", "a1", "a2"}
        for stats in out.player_stats.values():
            for values in stats.values():
                assert len(values) == len(out.home_scores)
        # Conservation must survive chunk concatenation across the whole run.
        allocated = sum(out.player_stats[p.player_id]["player_goal_scorer_anytime"] for p in HOME_SOCCER)
        assert np.array_equal(allocated, out.home_scores)

    def test_early_convergence_truncates_player_arrays(self) -> None:
        out = self.run(capture=True, iterations=10_000, convergence_threshold=100.0)
        assert out.converged
        assert out.iterations_run < 10_000
        for stats in out.player_stats.values():
            for values in stats.values():
                assert len(values) == out.iterations_run


class TestBuildPlayerDistributions:
    def make_output(self, player_stats: dict[str, dict[str, npt.NDArray[np.int32]]], n: int = 4000) -> SimulationOutput:
        home = np.full(n, 2, dtype=np.int32)
        away = np.full(n, 1, dtype=np.int32)
        return SimulationOutput(
            iterations_run=n,
            converged=True,
            convergence_iteration=n,
            standard_error=0.1,
            home_scores=home,
            away_scores=away,
            margins=home - away,
            totals=home + away,
            home_win_prob=1.0,
            away_win_prob=0.0,
            draw_prob=0.0,
            margin_mean=1.0,
            margin_std=0.0,
            total_mean=3.0,
            total_std=0.0,
            player_stats=player_stats,
        )

    ROSTER = {"h1": soccer_rates("h1", "HOME", 0.6), "a1": soccer_rates("a1", "AWAY", 0.4)}

    def test_yes_no_stat_emits_yes_probability_not_over_grid(self) -> None:
        rng = np.random.default_rng(1)
        goals = rng.poisson(0.8, 4000).astype(np.int32)
        players = build_player_distributions(
            self.make_output({"h1": {"player_goal_scorer_anytime": goals}}), self.ROSTER
        )
        block = players["h1"].stats["player_goal_scorer_anytime"]
        assert block.over_probabilities is None
        assert block.yes_probability == pytest.approx(float(np.mean(goals > 0)), abs=1e-4)

    def test_over_grid_is_monotonically_nonincreasing(self) -> None:
        rng = np.random.default_rng(2)
        shots = rng.poisson(2.4, 4000).astype(np.int32)
        players = build_player_distributions(self.make_output({"h1": {"player_shots": shots}}), self.ROSTER)
        block = players["h1"].stats["player_shots"]
        assert block.yes_probability is None
        assert block.over_probabilities is not None
        lines = sorted(float(line) for line in block.over_probabilities)
        probs = [block.over_probabilities[f"{line:.1f}"] for line in lines]
        assert all(a >= b for a, b in zip(probs, probs[1:], strict=False))
        assert all(not float(line).is_integer() for line in lines)  # half-point lines cannot push

    def test_distribution_block_matches_sample_moments(self) -> None:
        rng = np.random.default_rng(3)
        points = rng.poisson(21.0, 4000).astype(np.int32)
        players = build_player_distributions(self.make_output({"h1": {"player_points": points}}), self.ROSTER)
        dist = players["h1"].stats["player_points"].distribution
        assert dist.mean == pytest.approx(float(np.mean(points)), abs=0.01)
        assert dist.min == int(points.min())
        assert dist.max == int(points.max())
        assert sum(dist.values.values()) == pytest.approx(1.0, abs=0.01)

    def test_metadata_comes_from_roster(self) -> None:
        values = np.ones(100, dtype=np.int32)
        players = build_player_distributions(self.make_output({"a1": {"player_shots": values}}, n=100), self.ROSTER)
        assert players["a1"].name == "A1"
        assert players["a1"].team == "AWAY"

    def test_players_missing_from_roster_are_skipped(self) -> None:
        values = np.ones(100, dtype=np.int32)
        players = build_player_distributions(self.make_output({"ghost": {"player_shots": values}}, n=100), self.ROSTER)
        assert players == {}


class TestDefaultLineGrid:
    def test_half_point_lines_centered_on_mean(self) -> None:
        lines = default_line_grid(4.2)
        assert lines == [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

    def test_low_means_drop_negative_lines(self) -> None:
        lines = default_line_grid(0.4)
        assert lines == [0.5, 1.5, 2.5, 3.5]
        assert all(line > 0 for line in lines)
