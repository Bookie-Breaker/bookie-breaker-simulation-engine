"""Simulation orchestration: fetch stats, cache-check, run, persist, publish."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from datetime import UTC, datetime

import redis.asyncio as aioredis

from simulation_engine import __version__
from simulation_engine.api.errors import ApiError, DuplicateResourceError, NotFoundError, UnprocessableError
from simulation_engine.api.models import (
    BatchData,
    BatchGameRequest,
    BatchGameResult,
    BatchResultSummary,
    CorrelationsData,
    DistributionsData,
    DistributionType,
    HealthData,
    HealthLoad,
    LiveStateIn,
    PlayerDistributionsData,
    SimulationConfigIn,
    SimulationConfigOut,
    SimulationRunData,
)
from simulation_engine.cache.redis_cache import SimulationCache
from simulation_engine.clients.statistics import PlayerDetail, ProbablePitcher, StatisticsClient
from simulation_engine.config import Settings
from simulation_engine.core.correlations import CorrelationArtifact, UnknownLegError, build_correlation_artifact
from simulation_engine.core.hashing import compute_parameters_hash, compute_roster_signature
from simulation_engine.core.output import build_distributions, build_player_distributions, build_result
from simulation_engine.core.params import GameContext, LiveState, PlayerRates
from simulation_engine.core.player_rates import build_player_rates
from simulation_engine.core.plugins import get_plugin
from simulation_engine.core.runner import run_monte_carlo
from simulation_engine.events.publisher import publish_simulation_completed

logger = logging.getLogger(__name__)

_TERMINAL_GAME_STATUSES = frozenset({"FINAL", "CANCELLED"})
#: Concurrent player-detail fetches per roster resolution.
_ROSTER_FETCH_CONCURRENCY = 8


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _starter_fip(pitcher: ProbablePitcher | None) -> float | None:
    """FIP of an announced probable starter; None when unannounced or FIP is missing."""
    if pitcher is None or pitcher.fip <= 0:
        return None
    return pitcher.fip


def _hashable_config(config: SimulationConfigIn) -> dict[str, object]:
    """Config payload for hashing with the default include_player_props stripped.

    include_player_props=False (the default) is removed so pregame parameter
    hashes and idempotency body hashes stay byte-identical to pre-Wave-3
    hashes; True stays in the payload and yields a distinct hash (a
    props-enabled run stores different artifacts and must never be replayed
    from a props-off cache entry, or vice versa).
    """
    payload = config.model_dump(mode="json")
    if not config.include_player_props:
        payload.pop("include_player_props", None)
    return payload


def _request_body_hash(
    game_id: str, config: SimulationConfigIn, force_refresh: bool, live_state: LiveStateIn | None = None
) -> str:
    payload: dict[str, object] = {
        "game_id": game_id,
        "config": _hashable_config(config),
        "force_refresh": force_refresh,
    }
    # Included only when set so pregame body hashes (and thus in-flight
    # idempotency-key records) stay identical to pre-Wave-2 hashes.
    if live_state is not None:
        payload["live_state"] = live_state.model_dump(mode="json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


_VALID_BASES = frozenset({"---", "1--", "-2-", "--3", "12-", "1-3", "-23", "123"})
_VALID_HALVES = frozenset({"TOP", "BOTTOM"})
_VALID_POSSESSIONS = frozenset({"HOME", "AWAY"})


def _to_live_state(live: LiveStateIn) -> LiveState:
    """Validate a live-state request body and convert it to the domain dataclass.

    Bounds violations raise 422 UNPROCESSABLE_ENTITY per the Wave 2 contract
    (see LiveStateIn for why these are not pydantic Field constraints).
    """
    problems: list[str] = []
    if live.home_score < 0 or live.away_score < 0:
        problems.append("home_score and away_score must be >= 0")
    if not 0.0 < live.fraction_remaining <= 1.0:
        problems.append("fraction_remaining must be in (0, 1]")
    if live.period is not None and live.period < 1:
        problems.append("period must be >= 1")
    if live.clock_seconds is not None and live.clock_seconds < 0:
        problems.append("clock_seconds must be >= 0")
    if live.bases is not None and live.bases not in _VALID_BASES:
        problems.append(f"bases must be one of {sorted(_VALID_BASES)}")
    if live.outs is not None and not 0 <= live.outs <= 2:
        problems.append("outs must be between 0 and 2")
    if live.half is not None and live.half not in _VALID_HALVES:
        problems.append("half must be 'TOP' or 'BOTTOM'")
    if live.possession is not None and live.possession not in _VALID_POSSESSIONS:
        problems.append("possession must be 'HOME' or 'AWAY'")
    if live.down is not None and not 1 <= live.down <= 4:
        problems.append("down must be between 1 and 4")
    if live.yardline is not None and not 0 <= live.yardline <= 100:
        problems.append("yardline must be between 0 and 100")
    if problems:
        raise UnprocessableError(f"Invalid live_state: {'; '.join(problems)}")
    return LiveState(
        home_score=live.home_score,
        away_score=live.away_score,
        fraction_remaining=live.fraction_remaining,
        period=live.period,
        clock_seconds=live.clock_seconds,
        bases=live.bases,
        outs=live.outs,
        half=live.half,
        possession=live.possession,
        down=live.down,
        yardline=live.yardline,
    )


class SimulationService:
    def __init__(
        self,
        settings: Settings,
        cache: SimulationCache,
        statistics: StatisticsClient,
        redis_client: "aioredis.Redis",
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._statistics = statistics
        self._redis = redis_client
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_simulations)
        self._active = 0
        self._queued = 0
        self._recent_durations: deque[int] = deque(maxlen=100)
        self._started_monotonic = time.monotonic()

    async def run_simulation(
        self,
        game_id: str,
        config: SimulationConfigIn,
        force_refresh: bool = False,
        idempotency_key: str | None = None,
        live_state: LiveStateIn | None = None,
    ) -> SimulationRunData:
        live = _to_live_state(live_state) if live_state is not None else None
        body_hash = _request_body_hash(game_id, config, force_refresh, live_state)
        if idempotency_key is not None:
            existing = await self._cache.get_idempotent(idempotency_key)
            if existing is not None:
                stored_hash, stored_run_id = existing
                if stored_hash != body_hash:
                    raise DuplicateResourceError("X-Idempotency-Key was already used with a different request body")
                replayed = await self._cache.get_run(stored_run_id)
                if replayed is not None:
                    return replayed.model_copy(update={"cached": True})

        game = await self._statistics.get_game(game_id)
        if game.status in _TERMINAL_GAME_STATUSES:
            raise UnprocessableError(f"Game {game_id} is {game.status}; simulation is not applicable")

        spec = get_plugin(game.league)
        home_stats, away_stats = await asyncio.gather(
            self._statistics.get_team_stats(game.home_team.id),
            self._statistics.get_team_stats(game.away_team.id),
        )
        home_params = spec.map_team_stats(home_stats)
        away_params = spec.map_team_stats(away_stats)

        # Player props (Phase 7 Wave 3): resolve both rosters up front so the
        # roster signature participates in the parameters hash. Empty rosters
        # (dormant sports, stubbed providers) proceed WITHOUT player output;
        # only transport errors hard-fail.
        player_rates_home: list[PlayerRates] = []
        player_rates_away: list[PlayerRates] = []
        roster_signature: str | None = None
        if config.include_player_props:
            home_roster, away_roster = await asyncio.gather(
                self._fetch_roster(game.league, game.home_team.id),
                self._fetch_roster(game.league, game.away_team.id),
            )
            player_rates_home = build_player_rates(spec.label, home_roster, "HOME")
            player_rates_away = build_player_rates(spec.label, away_roster, "AWAY")
            if not player_rates_home and not player_rates_away:
                logger.warning(
                    "include_player_props requested for game %s (%s) but no usable roster data exists; "
                    "proceeding without player output",
                    game_id,
                    game.league,
                )
            roster_signature = compute_roster_signature(player_rates_home, player_rates_away)

        # Probable starters (BASEBALL leagues) enter the context — and thus
        # the parameters hash — so a starter announcement invalidates cached
        # simulations. The baseball plugin applies each starter to the
        # OPPOSING batting side (home batters face the away starter).
        # live_state (when set) enters the context — and thus the parameter
        # hash — so every distinct in-game state gets its own cache entry
        # while pregame (live_state=None) hashes stay byte-identical.
        # roster_signature (when props are on) does the same for rosters.
        context = GameContext(
            league=game.league,
            home_starter_fip=_starter_fip(game.home_probable_pitcher),
            away_starter_fip=_starter_fip(game.away_probable_pitcher),
            live_state=live,
            roster_signature=roster_signature,
        )
        config_dict = _hashable_config(config)
        parameters_hash = compute_parameters_hash(
            game_id, home_params, away_params, context, config_dict, plugin_label=spec.label
        )

        if not force_refresh:
            cached_run_id = await self._cache.get_cached_run_id(game_id, parameters_hash)
            if cached_run_id is not None:
                cached_run = await self._cache.get_run(cached_run_id)
                if cached_run is not None:
                    return cached_run.model_copy(update={"cached": True})

        simulator = spec.simulator({**spec.plugin_config, **config.plugin_config})
        if config.include_player_props:
            simulator.set_players(player_rates_home, player_rates_away)
        started_at = _utc_now_iso()
        self._queued += 1
        async with self._semaphore:
            self._queued -= 1
            self._active += 1
            try:
                output = await asyncio.to_thread(
                    run_monte_carlo,
                    simulator,
                    home_params,
                    away_params,
                    context,
                    config.iterations,
                    config.convergence_threshold,
                    self._settings.convergence_check_interval,
                    config.random_seed,
                    grid_config=spec.grid_config,
                    capture_players=config.include_player_props,
                )
            finally:
                self._active -= 1

        duration_ms = int(output.elapsed_ms)
        self._recent_durations.append(duration_ms)
        run = SimulationRunData(
            simulation_run_id=str(uuid.uuid4()),
            game_id=game_id,
            status="completed",
            cached=False,
            config=SimulationConfigOut(
                sport=simulator.get_sport(),
                iterations=config.iterations,
                convergence_threshold=config.convergence_threshold,
                random_seed=config.random_seed,
            ),
            started_at=started_at,
            completed_at=_utc_now_iso(),
            duration_ms=duration_ms,
            iterations_completed=output.iterations_run,
            converged=output.converged,
            parameters_hash=parameters_hash,
            result=build_result(output, str(uuid.uuid4())),
        )

        distributions = {name: dist.model_dump(mode="json") for name, dist in build_distributions(output).items()}
        # The raw per-iteration sample arrays are not retained past this point,
        # so the parlay-correlation artifact (including the packed boolean leg
        # matrix that makes arbitrary-subset joints computable at read time)
        # must be built here, from the in-memory output.
        correlations = build_correlation_artifact(
            output,
            include_draw=simulator.get_sport() == "SOCCER",
            joint_goal_grid=simulator.joint_grid(),
        ).to_payload()
        # Player distributions (Phase 7 Wave 3): built from the in-memory
        # arrays for the same reason as correlations — the raw samples are
        # not retained past this point. Stored whenever props were REQUESTED
        # (even with empty player output) so the read path can distinguish
        # "captured, no roster data" (200 with empty players) from "not
        # captured" (404).
        player_distributions: dict[str, object] | None = None
        if config.include_player_props:
            roster_index = {p.player_id: p for p in player_rates_home + player_rates_away}
            players = build_player_distributions(output, roster_index)
            player_distributions = {
                "game_id": game_id,
                "iterations_completed": output.iterations_run,
                "players": {player_id: entry.model_dump(mode="json") for player_id, entry in players.items()},
            }
        await self._cache.store_run(run, distributions, correlations, player_distributions)
        await publish_simulation_completed(self._redis, run, game.league)
        if idempotency_key is not None:
            await self._cache.store_idempotent(idempotency_key, body_hash, run.simulation_run_id)
        return run

    async def _fetch_roster(self, league: str, team_id: str) -> list[PlayerDetail]:
        """Resolve one team's roster to player details with bounded concurrency.

        Individual players that 404 mid-roster are skipped with a warning
        (stale roster entries must not fail the whole run); transport errors
        (DependencyError / DependencyTimeoutError) propagate and fail the
        request, per the Wave 3 contract.
        """
        summaries = await self._statistics.get_team_players(league, team_id)
        semaphore = asyncio.Semaphore(_ROSTER_FETCH_CONCURRENCY)

        async def fetch(player_id: str) -> PlayerDetail | None:
            async with semaphore:
                try:
                    return await self._statistics.get_player(player_id)
                except NotFoundError:
                    logger.warning("player %s listed on team %s roster but not found; skipping", player_id, team_id)
                    return None

        details = await asyncio.gather(*(fetch(summary.id) for summary in summaries))
        return [detail for detail in details if detail is not None]

    async def get_run(self, simulation_id: str) -> SimulationRunData:
        run = await self._cache.get_run(simulation_id)
        if run is None:
            raise NotFoundError(f"Simulation run {simulation_id} not found (results expire after 2 hours)")
        return run

    async def get_latest(self, game_id: str, force_refresh: bool = False) -> SimulationRunData:
        if force_refresh:
            return await self.run_simulation(game_id, self.default_config(), force_refresh=True)
        run_id = await self._cache.get_latest_run_id(game_id)
        if run_id is None:
            raise NotFoundError(f"No simulations found for game {game_id}")
        return await self.get_run(run_id)

    async def get_distributions(self, simulation_id: str, distribution_type: DistributionType) -> DistributionsData:
        run = await self.get_run(simulation_id)
        stored = await self._cache.get_distributions(run.game_id)
        if stored is None or stored.get("simulation_run_id") != simulation_id:
            raise NotFoundError(f"Distributions for simulation {simulation_id} are no longer available")
        names = ["home_score", "away_score", "margin", "total"] if distribution_type == "all" else [distribution_type]
        distributions = {name: stored[name] for name in names if name in stored}
        return DistributionsData.model_validate(
            {
                "simulation_run_id": simulation_id,
                "game_id": run.game_id,
                "iterations_completed": run.iterations_completed,
                "distributions": distributions,
            }
        )

    async def get_player_distributions(
        self,
        simulation_id: str,
        player_id: str | None = None,
        stat_type: str | None = None,
    ) -> PlayerDistributionsData:
        """Player stat distributions for a run (Phase 7 Wave 3).

        Mirrors the correlations read-path semantics: 404 when the run is no
        longer the latest for its game or player props were not captured.
        Optional filters narrow to one player and/or one canonical stat key;
        filters that match nothing 404 (unknown player, uncaptured stat).
        """
        run = await self.get_run(simulation_id)
        stored = await self._cache.get_player_distributions(run.game_id)
        if stored is None or stored.get("simulation_run_id") != simulation_id:
            raise NotFoundError(
                f"Player distributions for simulation {simulation_id} are not available "
                "(run without include_player_props, or superseded by a newer run)"
            )
        players = stored.get("players")
        players = dict(players) if isinstance(players, dict) else {}
        if player_id is not None:
            if player_id not in players:
                raise NotFoundError(f"Player {player_id} has no captured distributions in simulation {simulation_id}")
            players = {player_id: players[player_id]}
        if stat_type is not None:
            filtered = {
                pid: {**entry, "stats": {stat_type: entry["stats"][stat_type]}}
                for pid, entry in players.items()
                if isinstance(entry, dict) and stat_type in entry.get("stats", {})
            }
            if players and not filtered:
                raise NotFoundError(f"Stat {stat_type!r} was not captured in simulation {simulation_id}")
            players = filtered
        return PlayerDistributionsData.model_validate(
            {
                "simulation_run_id": simulation_id,
                "game_id": run.game_id,
                "iterations_completed": stored.get("iterations_completed", run.iterations_completed),
                "players": players,
            }
        )

    async def get_correlations(self, simulation_id: str, legs: list[str] | None = None) -> CorrelationsData:
        """Correlation artifact for a run; with ``legs``, a subset view plus their joint probability.

        Requested legs must be stored artifact legs or exact half-point
        complements — the raw sample arrays are gone after the run, so legs
        outside that vocabulary cannot be recomputed and yield a 422.
        """
        run = await self.get_run(simulation_id)
        stored = await self._cache.get_correlations(run.game_id)
        if stored is None or stored.get("simulation_run_id") != simulation_id:
            raise NotFoundError(f"Correlations for simulation {simulation_id} are no longer available")
        artifact = CorrelationArtifact.from_payload(stored)
        if legs is None:
            return CorrelationsData(
                simulation_run_id=simulation_id,
                game_id=run.game_id,
                iterations=artifact.iterations,
                legs=artifact.legs,
                marginals=artifact.marginals,
                matrix=artifact.matrix,
                joint_goal_grid=artifact.joint_goal_grid,
            )
        try:
            marginals, matrix, joint = artifact.subset(legs)
        except UnknownLegError as exc:
            raise UnprocessableError(str(exc)) from exc
        return CorrelationsData(
            simulation_run_id=simulation_id,
            game_id=run.game_id,
            iterations=artifact.iterations,
            legs=legs,
            marginals=marginals,
            matrix=matrix,
            joint_probability=joint,
            joint_goal_grid=artifact.joint_goal_grid,
        )

    def default_config(self) -> SimulationConfigIn:
        return SimulationConfigIn(
            iterations=min(self._settings.simulation_iterations, self._settings.max_iterations),
            convergence_threshold=self._settings.convergence_threshold,
        )

    async def run_batch(
        self,
        games: list[BatchGameRequest],
        default_config: SimulationConfigIn,
        force_refresh: bool = False,
    ) -> BatchData:
        # Live re-simulation is single-game only in v1 (Phase 7 Wave 2):
        # reject the whole batch up front rather than per-game so the caller
        # gets an unambiguous 422 instead of a partial batch.
        rejected = [entry.game_id for entry in games if entry.live_state is not None]
        if rejected:
            raise UnprocessableError(
                "live_state is not supported in batch simulations; "
                f"use POST /simulations per game (offending game_ids: {', '.join(rejected)})"
            )
        batch_id = str(uuid.uuid4())
        started_at = _utc_now_iso()
        started = time.perf_counter()

        async def run_one(entry: BatchGameRequest) -> BatchGameResult:
            try:
                run = await self.run_simulation(entry.game_id, entry.config or default_config, force_refresh)
            except ApiError as exc:
                return BatchGameResult(game_id=entry.game_id, status="failed", error=exc.message)
            return BatchGameResult(
                simulation_run_id=run.simulation_run_id,
                game_id=entry.game_id,
                status="completed",
                cached=run.cached,
                result=BatchResultSummary(
                    home_win_probability=run.result.home_win_probability,
                    away_win_probability=run.result.away_win_probability,
                    mean_total=run.result.mean_total,
                    mean_margin=run.result.mean_margin,
                ),
            )

        results = list(await asyncio.gather(*(run_one(entry) for entry in games)))
        failed = sum(1 for r in results if r.status == "failed")
        completed = len(results) - failed
        status: str = "completed" if failed == 0 else ("failed" if completed == 0 else "partial")
        batch = BatchData(
            batch_id=batch_id,
            status=status,  # type: ignore[arg-type]
            total_games=len(results),
            completed_games=completed,
            failed_games=failed,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            total_duration_ms=int((time.perf_counter() - started) * 1000),
            results=results,
        )
        await self._cache.store_batch(batch_id, batch.model_dump_json())
        return batch

    async def health(self) -> HealthData:
        redis_ok, stats_ok = await asyncio.gather(self._cache.is_healthy(), self._statistics.is_healthy())
        durations = list(self._recent_durations)
        return HealthData(
            status="healthy" if (redis_ok and stats_ok) else "degraded",
            version=__version__,
            uptime_seconds=int(time.monotonic() - self._started_monotonic),
            dependencies={
                "statistics_service": "healthy" if stats_ok else "unhealthy",
                "redis": "healthy" if redis_ok else "unhealthy",
            },
            load=HealthLoad(
                active_simulations=self._active,
                queued_simulations=self._queued,
                max_concurrent=self._settings.max_concurrent_simulations,
                simulations_today=await self._cache.simulations_today() if redis_ok else 0,
                avg_duration_ms=int(sum(durations) / len(durations)) if durations else 0,
            ),
        )
