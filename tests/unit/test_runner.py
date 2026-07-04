"""Monte Carlo runner tests using a deterministic fake simulator."""

import numpy as np
import numpy.typing as npt
import pytest

from simulation_engine.core.framework import GameResult, GameSimulator
from simulation_engine.core.params import GameContext, TeamParams
from simulation_engine.core.runner import run_monte_carlo


class FakeSimulator(GameSimulator):
    """Draws scores from normal distributions; convergence-friendly."""

    def __init__(self, home_mean: float = 112.0, away_mean: float = 108.0, std: float = 10.0) -> None:
        self._home_mean = home_mean
        self._away_mean = away_mean
        self._std = std

    def set_parameters(self, home_params: TeamParams, away_params: TeamParams, context: GameContext) -> None:
        pass

    def simulate_game(self, rng: np.random.Generator) -> GameResult:
        home, away = self.simulate_games(rng, 1)
        return GameResult(home_score=int(home[0]), away_score=int(away[0]))

    def simulate_games(self, rng: np.random.Generator, n: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        home = np.rint(rng.normal(self._home_mean, self._std, n)).astype(np.int32)
        away = np.rint(rng.normal(self._away_mean, self._std, n)).astype(np.int32)
        return home, away

    def get_sport(self) -> str:
        return "BASKETBALL"

    def get_league(self) -> str:
        return "NBA"


def run(simulator: FakeSimulator, make_team_params, **kwargs):
    return run_monte_carlo(simulator, make_team_params("h"), make_team_params("a"), GameContext(), **kwargs)


class TestRunner:
    def test_runs_requested_iterations(self, make_team_params) -> None:
        # Wide threshold=0 disables SE criterion; identical consecutive
        # distributions still trigger stability, so use few iterations
        out = run(FakeSimulator(), make_team_params, iterations=3000, convergence_threshold=1e-9, seed=1)
        assert out.iterations_run == 3000
        assert len(out.home_scores) == 3000
        assert out.elapsed_ms > 0

    def test_early_stop_truncates_arrays(self, make_team_params) -> None:
        # A huge SE threshold converges at the first check (2000 iterations)
        out = run(FakeSimulator(), make_team_params, iterations=10_000, convergence_threshold=100.0, seed=1)
        assert out.converged
        assert out.convergence_iteration == 2000
        assert out.iterations_run == 2000
        assert len(out.margins) == 2000

    def test_standard_error_always_reported(self, make_team_params) -> None:
        out = run(FakeSimulator(), make_team_params, iterations=3000, convergence_threshold=1e-9, seed=1)
        expected_se = float(np.std(out.margins, ddof=1) / np.sqrt(len(out.margins)))
        assert out.standard_error == pytest.approx(expected_se)

    def test_probabilities_are_consistent(self, make_team_params) -> None:
        out = run(FakeSimulator(), make_team_params, iterations=4000, convergence_threshold=1e-9, seed=3)
        assert out.home_win_prob + out.away_win_prob + out.draw_prob == pytest.approx(1.0)
        assert out.home_win_prob > 0.5  # home mean is 4 points higher

    def test_spread_cover_semantics(self, make_team_params) -> None:
        out = run(
            FakeSimulator(),
            make_team_params,
            iterations=4000,
            convergence_threshold=1e-9,
            seed=3,
            common_spreads=[-3.5, +3.5],
        )
        # home -3.5 covers when margin > 3.5; home +3.5 covers when margin > -3.5
        assert out.spread_covers[-3.5] == pytest.approx(float(np.mean(out.margins > 3.5)))
        assert out.spread_covers[3.5] == pytest.approx(float(np.mean(out.margins > -3.5)))
        assert out.spread_covers[3.5] > out.spread_covers[-3.5]

    def test_default_grids_cover_means(self, make_team_params) -> None:
        out = run(FakeSimulator(), make_team_params, iterations=4000, convergence_threshold=1e-9, seed=3)
        spread_lines = sorted(out.spread_covers)
        total_lines = sorted(out.total_overs)
        assert spread_lines[0] < -out.margin_mean < spread_lines[-1]
        assert total_lines[0] < out.total_mean < total_lines[-1]
        # over probabilities decrease as the line rises
        over_probs = [out.total_overs[line] for line in total_lines]
        assert all(a >= b for a, b in zip(over_probs, over_probs[1:], strict=False))


class TestSeedReproducibility:
    def test_same_seed_identical_results(self, make_team_params) -> None:
        a = run(FakeSimulator(), make_team_params, iterations=3000, convergence_threshold=1e-9, seed=42)
        b = run(FakeSimulator(), make_team_params, iterations=3000, convergence_threshold=1e-9, seed=42)
        assert np.array_equal(a.home_scores, b.home_scores)
        assert np.array_equal(a.away_scores, b.away_scores)
        assert a.home_win_prob == b.home_win_prob

    def test_different_seed_differs(self, make_team_params) -> None:
        a = run(FakeSimulator(), make_team_params, iterations=3000, convergence_threshold=1e-9, seed=42)
        b = run(FakeSimulator(), make_team_params, iterations=3000, convergence_threshold=1e-9, seed=43)
        assert not np.array_equal(a.home_scores, b.home_scores)
