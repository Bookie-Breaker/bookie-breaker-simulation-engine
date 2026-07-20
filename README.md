# bookie-breaker-simulation-engine

[![CI](https://img.shields.io/github/actions/workflow/status/Bookie-Breaker/bookie-breaker-simulation-engine/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white)](https://github.com/Bookie-Breaker/bookie-breaker-simulation-engine/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/codecov/c/github/Bookie-Breaker/bookie-breaker-simulation-engine?logo=codecov&logoColor=white)](https://app.codecov.io/gh/Bookie-Breaker/bookie-breaker-simulation-engine)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

Sport-agnostic Monte Carlo engine (port 8003) producing score, margin, and total distributions
for sports matchups via pluggable per-sport simulators. Player-level distributions and
correlation outputs feed the prop and parlay math downstream. Results are cached in Redis —
there is no database — and simulations draw their inputs from the statistics-service.
`SIMULATION_ITERATIONS` (default 10000) trades distribution depth for speed.

## Quickstart

### With Docker Compose (recommended)

```bash
task up  # from BookieBreaker/ root
```

### Standalone

```bash
cp .env.example .env  # fill in values
task bootstrap
task dev
```

## API

Interactive docs at `http://localhost:8003/docs` when running. All endpoints live under
`/api/v1/sim`.

Full contract:
[simulation-engine-api.md](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/api-contracts/simulation-engine-api.md)

## Architecture Decisions

- [Sport-Agnostic Framework (ADR-001)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/001-sport-agnostic-framework.md)
- [Hybrid Prediction Approach (ADR-002)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/002-hybrid-prediction-approach.md)
- [Football Simulation Granularity (ADR-018)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/018-football-simulation-granularity.md)
- [Sport Expansion Scope and Data Sources (ADR-026)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/026-sport-expansion-scope-and-data-sources.md)
- [Parlay Joint Probability and Correlated Kelly (ADR-030)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/030-parlay-joint-probability-and-correlated-kelly.md)

How simulations fit into pipeline runs:
[Pipeline and Scheduling playbook](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/playbooks/05-pipeline-and-scheduling.md)

## Environment Variables

See `.env.example` for all variables with descriptions. Key ones: `STATISTICS_SERVICE_URL`,
`REDIS_URL`, `SIMULATION_ITERATIONS` (default 10000), `PORT=8003`.
