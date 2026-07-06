"""League-average constants used by the simulation plugins.

NBA values are fallbacks for basketball plugin parameters the
statistics-service Phase 1 contract does not expose (see
algorithms/simulation-algorithms.md section 3 input table vs
api-contracts/statistics-service-api.md). Values are recent NBA league-wide
season averages; refresh occasionally or replace once the statistics-service
contract grows the missing fields.

Soccer values (Phase 6 Wave 1, ADR-026) are per-competition priors for the
Dixon-Coles Poisson plugin.
"""

NBA_LEAGUE_AVG_PACE = 100.0
NBA_THREE_ATTEMPT_RATE = 0.39  # 3PA / FGA
NBA_FT_RATE = 0.26  # FTA / FGA
NBA_TOV_PCT = 13.0  # turnovers per 100 possessions
NBA_OREB_PCT = 27.0  # offensive rebound percentage
NBA_THREE_PCT = 0.36
NBA_FT_PCT = 0.78
NBA_AND_ONE_RATE = 0.03  # and-1 fouls on made two-point baskets
NBA_HOME_ADVANTAGE = 1.5  # points per 100 possessions
NBA_POSSESSION_STD = 3.0  # game-to-game std dev of possessions
NBA_OT_POSSESSION_FRACTION = 5.0 / 48.0  # overtime length relative to regulation

# Soccer (Phase 6 Wave 1). Tunable priors; validated against real data in the
# verification session.
SOCCER_WC_BASE_GOALS_PER_TEAM = 1.35  # World Cup goals per team per match
SOCCER_EPL_BASE_GOALS_PER_TEAM = 1.45  # EPL goals per team per match
SOCCER_EPL_HOME_GOAL_MULTIPLIER = 1.15  # club home advantage on the home goal rate
SOCCER_DC_RHO = -0.11  # Dixon-Coles low-score dependence parameter
