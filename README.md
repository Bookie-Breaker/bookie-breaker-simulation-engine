# bookie-breaker-simulation-engine

Monte Carlo simulation engine producing score, margin, and total distributions for sports matchups.

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

API documentation available at `http://localhost:8003/docs` when running.

## Architecture Decisions

- [Tech Stack Selection (ADR-010)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/010-tech-stack-selection.md)

## Environment Variables

See `.env.example` for all variables with descriptions.
