"""Shared joint-Poisson score grid with the Dixon-Coles low-score correction.

Factored out of the soccer plugin (Phase 6 Wave 4) so hockey can reuse the
same machinery: two independent Poisson marginals form a joint score PMF grid,
the Dixon-Coles tau correction rescales the four low-score cells, and games
are sampled as one vectorized categorical draw over the flattened grid.
Soccer uses a 13x13 grid with rho = -0.11; hockey a 10x10 grid with a milder
rho = -0.05.
"""

import math

import numpy as np
import numpy.typing as npt


def poisson_pmf(lam: float, grid_size: int) -> npt.NDArray[np.float64]:
    """Poisson PMF over 0..grid_size-1 via the stable multiplicative recurrence."""
    pmf = np.empty(grid_size, dtype=np.float64)
    pmf[0] = math.exp(-lam)
    for k in range(1, grid_size):
        pmf[k] = pmf[k - 1] * lam / k
    return pmf


def build_goal_grid(lam_home: float, lam_away: float, rho: float, grid_size: int) -> npt.NDArray[np.float64]:
    """Joint score PMF: independent Poissons with the Dixon-Coles tau correction.

    Rows are home goals, columns are away goals. The tau correction rescales
    the four low-score cells (with lambda = lam_home, mu = lam_away):
    tau(0,0) = 1 - lambda*mu*rho, tau(1,0) = 1 + mu*rho,
    tau(0,1) = 1 + lambda*rho, tau(1,1) = 1 - rho. Any cell driven negative is
    clamped to zero and the grid is renormalized to sum to exactly 1 (the
    renormalization also absorbs the truncated Poisson tail mass).
    """
    grid = np.outer(poisson_pmf(lam_home, grid_size), poisson_pmf(lam_away, grid_size))
    grid[0, 0] *= 1.0 - lam_home * lam_away * rho
    grid[1, 0] *= 1.0 + lam_away * rho
    grid[0, 1] *= 1.0 + lam_home * rho
    grid[1, 1] *= 1.0 - rho
    np.clip(grid, 0.0, None, out=grid)
    normalized: npt.NDArray[np.float64] = grid / grid.sum()
    return normalized


def sample_grid(
    rng: np.random.Generator, cdf: npt.NDArray[np.float64], grid_size: int, n: int
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """One vectorized categorical draw of n (home, away) scores from a flattened grid CDF."""
    indices = np.searchsorted(cdf, rng.random(n), side="right")
    home_scores, away_scores = np.divmod(indices, grid_size)
    return home_scores.astype(np.int32), away_scores.astype(np.int32)
