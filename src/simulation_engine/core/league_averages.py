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

# Baseball (Phase 6 Wave 2). Documented-tunable priors for the half-inning
# runs plugin; validated against real data in the verification session.
MLB_RUNS_PER_GAME = 4.5  # league runs per team per game
NCAA_BSB_RUNS_PER_GAME = 6.5  # college baseball scores higher (dormant league)
MLB_LEAGUE_FIP = 4.10  # league-average FIP, starter multiplier baseline
MLB_LEAGUE_ERA = 4.20  # league-average ERA, bullpen multiplier baseline
BASEBALL_P0_BASE = 0.73  # scoreless half-inning probability at a league-average scoring rate
BASEBALL_P0_ALPHA = 0.35  # zero-inflation slope vs the target scoring rate (see plugins/baseball.py)

# Football (Phase 6 Wave 3, ADR-018). Documented-tunable priors for the
# drive-based plugin; validated against real data in the verification session.
NFL_DRIVES_PER_TEAM_MU = 10.9  # offensive drives per team per game
NFL_DRIVES_SIGMA = 1.2  # game-to-game std dev of drives per team
NFL_DRIVES_CLIP_MIN = 7
NFL_DRIVES_CLIP_MAX = 16
NCAA_FB_DRIVES_PER_TEAM_MU = 12.5  # college plays faster with more possessions
NCAA_FB_DRIVES_SIGMA = 1.6  # and with more game-to-game pace variance
NCAA_FB_DRIVES_CLIP_MIN = 8
NCAA_FB_DRIVES_CLIP_MAX = 18
NFL_POINTS_PER_DRIVE = 1.95  # league points per offensive drive
NCAA_FB_POINTS_PER_DRIVE = 2.15  # college scores more per possession
FOOTBALL_TD_FG_RATIO_BASE = 1.35  # p_td / p_fg at a league-average points-per-drive target
FOOTBALL_TD_FG_RATIO_ALPHA = 1.1  # ratio slope vs the target (see plugins/football.py band analysis)
FOOTBALL_TD_FG_RATIO_MIN = 0.05
FOOTBALL_TD_FG_RATIO_MAX = 4.0
NFL_HFA_MARGIN_POINTS = 2.2  # expected home margin shift; split +/-1.1 across each side's drives
NCAA_FB_HFA_MARGIN_POINTS = 3.0  # college crowds move lines further
FOOTBALL_OT_TARGET_MULTIPLIER = 1.4  # overtime drives score more (short fields, four-down urgency)

# Hockey (Phase 6 Wave 4, ADR-026). Documented-tunable priors for the Poisson
# grid plugin with OT/SO resolution.
NHL_GOALS_PER_TEAM = 3.0  # league goals per team per game
NHL_HOME_GOAL_MULT = 1.05  # home-ice multiplier on the home goal rate
HOCKEY_PP_WEIGHT = 0.5  # sensitivity of the goal rate to own power-play % vs league
HOCKEY_PK_WEIGHT = 0.5  # sensitivity of the goal rate to the opponent penalty-kill % vs league
NHL_LEAGUE_PP_PCT = 0.21  # league-average power-play conversion
NHL_LEAGUE_PK_PCT = 0.79  # league-average penalty-kill rate
NHL_LEAGUE_SAVE_PCT = 0.905  # league-average team save %; carried on params for hashing/debugging
HOCKEY_DC_RHO = -0.05  # milder low-score correction than soccer's -0.11
NHL_OT_SHARE_OF_TIES = 0.5  # ~half of regulation ties end in OT, the rest reach the shootout

# NCAA basketball (Phase 6 Wave 5, config-only reuse of the NBA simulator).
# Tunable college priors: slower 40-minute games, noisier pace, stronger home
# crowds. Shooting constants are carried for a future college-aware stats
# mapper; empty stat blocks currently fall back to the NBA constants above.
NCAA_BB_LEAGUE_AVG_PACE = 68.0  # possessions per 40 minutes
NCAA_BB_POSSESSION_STD = 4.0  # higher game-to-game pace variance than the NBA's 3.0
NCAA_BB_HOME_ADVANTAGE = 3.0  # points per 100 possessions; roughly double the NBA edge
NCAA_BB_POSSESSION_CLIP_MIN = 55
NCAA_BB_POSSESSION_CLIP_MAX = 100
NCAA_BB_OT_POSSESSION_FRACTION = 5.0 / 40.0  # 5-minute OT over a 40-minute regulation
NCAA_BB_THREE_PCT = 0.34  # college shooting runs below NBA percentages
NCAA_BB_FT_PCT = 0.72
NCAA_BB_THREE_ATTEMPT_RATE = 0.38
