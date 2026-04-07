# bookie-breaker-simulation-engine

## Service Purpose

Python/FastAPI Monte Carlo simulation engine producing score, margin, and total distributions for sports matchups. Sport-agnostic framework with pluggable sport-specific simulators.

## Language & Conventions

- **Language:** Python 3.12
- **Framework:** FastAPI + uvicorn
- **Project layout:** `src/simulation_engine/` package, `main.py` FastAPI entry point
- **Naming:** `snake_case.py` files, `snake_case` functions, `PascalCase` classes
- **Package manager:** uv
- **Testing:** pytest in `tests/`

## Key Files

- `src/simulation_engine/main.py` — FastAPI app entry point
- `src/simulation_engine/api/` — Route handlers
- `src/simulation_engine/core/` — Simulation framework and sport plugins
- `pyproject.toml` — Dependencies and tool config
- `.config/mise.toml` — Tool versions
- `.config/lefthook.yml` — Git hooks

## Service-Specific Commands

```bash
task dev          # uvicorn with --reload on port 8003
task lint         # ruff check + format
task test         # pytest --cov
task typecheck    # mypy src/
```

## Dependencies

- **statistics-service** (port 8002) — Team/player stats for simulation input
- **Redis** — Caching simulation results (keyed by parameters hash, 2h TTL)
- No database

## Environment Variables

See `.env.example`. Key: `STATISTICS_SERVICE_URL`, `REDIS_URL`, `SIMULATION_ITERATIONS=10000`, `PORT=8003`.
