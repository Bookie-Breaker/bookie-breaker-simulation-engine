"""NBA league-average constants.

Fallbacks for basketball plugin parameters the statistics-service Phase 1
contract does not expose (see algorithms/simulation-algorithms.md section 3
input table vs api-contracts/statistics-service-api.md). Values are recent
NBA league-wide season averages; refresh occasionally or replace once the
statistics-service contract grows the missing fields.
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
