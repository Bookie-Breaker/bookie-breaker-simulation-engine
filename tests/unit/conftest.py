"""Shared fixtures for unit tests."""

import pytest

from simulation_engine.core.params import TeamParams


@pytest.fixture
def make_team_params():
    def _make(team_id: str = "team", **overrides: float) -> TeamParams:
        defaults = {
            "pace": 100.0,
            "off_rating": 114.0,
            "def_rating": 112.0,
            "three_pct": 0.365,
            "two_pct": 0.54,
            "ft_pct": 0.78,
            "three_attempt_rate": 0.39,
            "ft_rate": 0.26,
            "tov_pct": 13.0,
            "oreb_pct": 27.0,
            "opp_three_pct": 0.36,
            "opp_two_pct": 0.54,
            "opp_ft_rate": 0.26,
            "forced_tov_pct": 13.0,
            "opp_oreb_pct": 27.0,
        }
        defaults.update(overrides)
        return TeamParams(team_id=team_id, abbreviation=team_id.upper()[:3], **defaults)  # type: ignore[arg-type]

    return _make
