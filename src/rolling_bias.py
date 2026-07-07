"""Rolling EWMA per-city model bias from forecast residuals (no lookahead)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESIDUALS_PATH = PROJECT_ROOT / "data" / "polymarket" / "rolling_bias_residuals.parquet"
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "polymarket" / "rolling_bias.json"


def ewma_bias(
    dated_residuals: list[tuple[str, float]],
    halflife_days: int,
) -> float:
    """EWMA over (date_str, residual) pairs sorted by date ascending."""
    if not dated_residuals or halflife_days <= 0:
        return 0.0
    ordered = sorted(dated_residuals, key=lambda x: x[0])
    alpha = 1.0 - np.exp(np.log(0.5) / halflife_days)
    ewma = float(ordered[0][1])
    for _, residual in ordered[1:]:
        ewma = alpha * float(residual) + (1.0 - alpha) * ewma
    return float(ewma)


def _assert_no_lookahead(rows: pd.DataFrame, event_date: str) -> None:
    if rows.empty:
        return
    max_date = str(rows["date"].max())
    if max_date >= event_date:
        raise ValueError(
            f"Lookahead in rolling bias: max residual date {max_date} >= event {event_date}"
        )


def load_residuals_df() -> pd.DataFrame:
    if not RESIDUALS_PATH.exists():
        return pd.DataFrame(columns=["city", "date", "forecast", "wu_actual", "residual"])
    df = pd.read_parquet(RESIDUALS_PATH)
    df["city"] = df["city"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def compute_rolling_bias(
    city: str,
    event_date: str,
    halflife_days: int = 20,
    min_obs: int = 5,
    max_correction_f: float = 1.5,
    residuals_df: pd.DataFrame | None = None,
) -> float:
    """EWMA of (forecast - wu_actual) for dates strictly before event_date."""
    df = residuals_df if residuals_df is not None else load_residuals_df()
    if df.empty:
        return 0.0
    sub = df[(df["city"] == city) & (df["date"] < event_date)].copy()
    _assert_no_lookahead(sub, event_date)
    if len(sub) < min_obs:
        return 0.0
    pairs = list(zip(sub["date"].astype(str), sub["residual"].astype(float)))
    raw_ewma = ewma_bias(pairs, halflife_days)
    return float(np.clip(raw_ewma, -max_correction_f, max_correction_f))


def write_snapshot(
    residuals_df: pd.DataFrame,
    halflife_days: int = 20,
    min_obs: int = 5,
    as_of_date: str | None = None,
) -> dict[str, dict[str, float | int | str]]:
    """Build per-city EWMA snapshot using all residuals up to as_of_date."""
    if residuals_df.empty:
        return {}
    df = residuals_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if as_of_date:
        df = df[df["date"] <= as_of_date]
    out: dict[str, dict[str, float | int | str]] = {}
    for city, grp in df.groupby("city"):
        grp = grp.sort_values("date")
        pairs = list(zip(grp["date"].astype(str), grp["residual"].astype(float)))
        n_obs = len(pairs)
        ewma = ewma_bias(pairs, halflife_days) if n_obs >= min_obs else 0.0
        out[str(city)] = {
            "ewma": round(ewma, 4),
            "n_obs": int(n_obs),
            "as_of_date": str(grp["date"].iloc[-1]) if n_obs else "",
        }
    return out


def save_residuals_and_snapshot(
    residuals_df: pd.DataFrame,
    halflife_days: int = 20,
    min_obs: int = 5,
) -> None:
    RESIDUALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = residuals_df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_parquet(RESIDUALS_PATH, index=False)
    as_of = str(out["date"].max()) if not out.empty else None
    snapshot = write_snapshot(out, halflife_days=halflife_days, min_obs=min_obs, as_of_date=as_of)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)


@dataclass
class RollingBiasCache:
    """Sequential backtest cache: record settled days, bias before trading."""

    halflife_days: int = 20
    min_obs: int = 5
    max_correction_f: float = 1.5
    _rows: list[dict[str, float | str]] = field(default_factory=list)

    def seed_from_parquet(self, path: Path | None = None) -> None:
        df = load_residuals_df() if path is None else pd.read_parquet(path)
        if df.empty:
            return
        for row in df.itertuples(index=False):
            self._rows.append(
                {
                    "city": str(row.city),
                    "date": str(row.date),
                    "forecast": float(row.forecast),
                    "wu_actual": float(row.wu_actual),
                    "residual": float(row.residual),
                }
            )

    def record(self, city: str, date_str: str, forecast: float, wu_actual: float) -> None:
        self._rows.append(
            {
                "city": city,
                "date": date_str,
                "forecast": float(forecast),
                "wu_actual": float(wu_actual),
                "residual": float(forecast) - float(wu_actual),
            }
        )

    def as_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["city", "date", "forecast", "wu_actual", "residual"])
        return pd.DataFrame(self._rows)

    def bias(self, city: str, event_date: str) -> float:
        df = self.as_dataframe()
        return compute_rolling_bias(
            city,
            event_date,
            halflife_days=self.halflife_days,
            min_obs=self.min_obs,
            max_correction_f=self.max_correction_f,
            residuals_df=df,
        )
