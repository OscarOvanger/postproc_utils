"""Build climatological prior table for Sequential Bayesian Kalman filter."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trackj.build_climatological_prior import (  # noqa: E402
    CITIES,
    CLI_END,
    CLI_START,
    build_climatological_prior,
    print_summary_table,
    run_sanity_checks,
)


def main() -> None:
    prior = build_climatological_prior(CITIES, CLI_START, CLI_END)
    print_summary_table(prior)
    run_sanity_checks(prior)


if __name__ == "__main__":
    main()
