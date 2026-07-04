"""Export the FastAPI-generated OpenAPI spec as deterministic YAML (ADR-021).

The generated spec is the source of truth for this service's API; the
exported artifact is committed to bookie-breaker-docs/api-contracts/openapi/.

Usage:
    uv run python scripts/export_openapi.py [--out PATH]
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simulation_engine.config import Settings  # noqa: E402
from simulation_engine.main import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    app = create_app(Settings())
    spec = app.openapi()
    rendered = yaml.safe_dump(spec, sort_keys=True, allow_unicode=True, default_flow_style=False)

    if args.out is None:
        sys.stdout.write(rendered)
    else:
        args.out.write_text(rendered)
        sys.stderr.write(f"wrote {args.out}\n")


if __name__ == "__main__":
    main()
