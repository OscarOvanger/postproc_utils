"""Train NGBoost distributional models for all cities and lead times."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ngboost_model import (  # noqa: E402
    CITIES,
    LEAD_TIMES,
    estimate_lead_correlation,
    preflight_feature_parquets,
    train_city_lead,
)


def _print_summary_table(summaries: list[dict]) -> None:
    header = (
        f"{'city':18s} | {'lead':4s} | {'n_train':>7s} | {'n_val':>5s} | "
        f"{'val_MAE':>7s} | {'val_CRPS':>8s} | {'mean_sigma':>10s} | {'n_est':>5s}"
    )
    print("\n=== TRAINING SUMMARY ===")
    print(header)
    print("-" * len(header))
    for row in summaries:
        val_mae = row.get("val_mae")
        val_crps = row.get("val_crps")
        mean_sigma = row.get("val_mean_sigma")
        print(
            f"{row['city']:18s} | {row['lead_time']:4s} | {row['n_train']:7d} | "
            f"{row['n_val']:5d} | "
            f"{f'{val_mae:.3f}' if val_mae is not None else 'n/a':>7s} | "
            f"{f'{val_crps:.3f}' if val_crps is not None else 'n/a':>8s} | "
            f"{f'{mean_sigma:.3f}' if mean_sigma is not None else 'n/a':>10s} | "
            f"{row['best_n_estimators']:5d}"
        )


def _print_interpretive_notes(summaries: list[dict]) -> None:
    print("\n=== INTERPRETIVE NOTES ===")
    by_city: dict[str, list[dict]] = {city: [] for city in CITIES}
    for row in summaries:
        by_city[row["city"]].append(row)

    for city in CITIES:
        rows = sorted(by_city[city], key=lambda r: r["lead_time"])
        maes = [r.get("val_mae") for r in rows if r.get("val_mae") is not None]
        sigmas = [r.get("val_mean_sigma") for r in rows if r.get("val_mean_sigma") is not None]
        if len(maes) >= 2:
            decreasing_mae = all(maes[i] >= maes[i + 1] for i in range(len(maes) - 1))
            print(
                f"  {city}: val_MAE t1→t3 = {maes} "
                f"({'decreasing' if decreasing_mae else 'NOT decreasing'})"
            )
        if len(sigmas) >= 2:
            decreasing_sigma = all(sigmas[i] >= sigmas[i + 1] for i in range(len(sigmas) - 1))
            print(
                f"  {city}: mean_sigma t1→t3 = {sigmas} "
                f"({'decreasing' if decreasing_sigma else 'NOT decreasing'})"
            )
        t2 = next((r for r in rows if r["lead_time"] == "t2"), None)
        if t2 and t2.get("val_mae") is not None:
            print(
                f"  {city}: t2 val_MAE={t2['val_mae']:.3f} "
                "(compare to Track-B test MAE when metrics available)"
            )

    print(
        "  Track-B comparison deferred: no local Track-B metrics found. "
        "Typical good daily Tmax MAE is ~2–3°F."
    )


def _print_correlation_table(corr_rows: list[dict]) -> None:
    header = (
        f"{'city':18s} | {'R[0,1]':>7s} | {'R[0,2]':>7s} | {'R[1,2]':>7s} | "
        f"{'r2_1':>6s} | {'r2_2':>6s} | {'r2_3':>6s}"
    )
    print("\n=== CROSS-LEAD CORRELATION (2025 val) ===")
    print(header)
    print("-" * len(header))
    for row in corr_rows:
        print(
            f"{row['city']:18s} | {row['R_01']:7.4f} | {row['R_02']:7.4f} | "
            f"{row['R_12']:7.4f} | {row['r2_1']:6.4f} | {row['r2_2']:6.4f} | {row['r2_3']:6.4f}"
        )


def main() -> None:
    missing = preflight_feature_parquets()
    if missing:
        print("Feature parquets not ready for training. Missing or undersized:")
        for item in missing:
            print(f"  - {item}")
        print(
            "\nWait for scripts/build_ngboost_features.py to finish, then re-run:\n"
            "  .venv/bin/python scripts/train_ngboost_all.py"
        )
        sys.exit(1)

    print("=== NGBoost Training: 6 cities × 3 lead times ===\n")
    summaries: list[dict] = []
    for city in CITIES:
        print(f"--- {city} ---")
        for lead_time in LEAD_TIMES:
            summaries.append(train_city_lead(city, lead_time))

    _print_summary_table(summaries)
    _print_interpretive_notes(summaries)

    corr_rows: list[dict] = []
    for city in CITIES:
        if city == "austin":
            print(f"\nSkipping correlation for austin (no 2025 validation dates).")
            continue
        result = estimate_lead_correlation(city)
        if result is None:
            print(f"\nSkipping correlation for {city} (insufficient 2025 overlap).")
            continue
        corr_rows.append(result)

    if corr_rows:
        _print_correlation_table(corr_rows)
        print("\nCorrelation saved to data/ngboost/correlation/{city}_R.npy and {city}_innov.json")


if __name__ == "__main__":
    main()
