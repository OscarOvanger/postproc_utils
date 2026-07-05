#!/usr/bin/env python3
"""Two diagnostic tests: hit-rate metric parity and settlement anchoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.common import (  # noqa: E402
    POLY_CITIES,
    load_day_snapshot,
    load_wu_targets,
    quotes_at_entry,
    select_entry_snapshot,
    has_entry_window_snapshot,
)
from scan_modal_buckets import compute_midpoint  # noqa: E402
from train_ngboost import (  # noqa: E402
    STATION_META,
    TARGET,
    apply_saved_median_fill,
    apply_sigma_calibration,
    load_saved_artifacts,
    predict_dist_params,
    temporal_split,
    transform_features,
)

MODEL_DIR = PROJECT_ROOT / "models" / "ngboost_v2"
REPORT_JSON = PROJECT_ROOT / "reports" / "fable_diagnostics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "fable_diagnostics.md"


def bucket_lower_bound(t: float) -> int:
    """Lower bound of the 2°F Polymarket bucket containing temperature t."""
    ti = int(t)
    return ti if ti % 2 == 0 else ti - 1


def bucket_index_from_temp(t: float) -> int:
    """Map temperature to sorted bucket index (0 = lowest bucket in snapshot order)."""
    lo = bucket_lower_bound(t)
    return lo


def bucket_index_from_label(label: str) -> int | None:
    """Sort key / index for a snapshot bucket label."""
    import re

    text = str(label).strip()
    le = re.match(r"^<=(\d+)$", text)
    if le:
        return int(le.group(1)) - 1000  # sort before range buckets
    ge = re.match(r"^>=(\d+)$", text)
    if ge:
        return int(ge.group(1)) + 1000  # sort after range buckets
    rng = re.match(r"^(\d+)-(\d+)$", text)
    if rng:
        return int(rng.group(1))
    return None


def bucket_midpoint_for_implied_mean(label: str) -> float | None:
    """Midpoint temperature for market-implied mean calculation."""
    import re

    text = str(label).strip()
    le = re.match(r"^<=(\d+)$", text)
    if le:
        return float(int(le.group(1))) - 2.0
    ge = re.match(r"^>=(\d+)$", text)
    if ge:
        return float(int(ge.group(1))) + 2.0
    rng = re.match(r"^(\d+)-(\d+)$", text)
    if rng:
        lower, upper = int(rng.group(1)), int(rng.group(2))
        return (lower + upper) / 2.0 + 0.5
    return None


def load_test_predictions() -> pd.DataFrame:
    """Load NGBoost v2 predictions on the 2026 test set."""
    model, scaler, config = load_saved_artifacts(MODEL_DIR)
    cities = list(config.get("cities", sorted(STATION_META.keys())))
    feature_cols = list(config.get("feature_columns", []))
    fill_medians = dict(config.get("nan_fill_medians", {}))

    from train_ngboost import assemble_dataset, drop_incomplete_rows

    stage1_path = MODEL_DIR / config.get("stage1_model", "lgb_stage1.pkl")
    lgb_model = joblib.load(stage1_path)
    stage1_cols = [c for c in feature_cols if c != "lgb_tmax_pred"]

    df = assemble_dataset(cities)
    df = drop_incomplete_rows(df)
    _train, _val, test_df = temporal_split(df)
    fill_cols = list(fill_medians.keys()) if fill_medians else []
    test_df = apply_saved_median_fill(test_df, fill_medians, fill_cols)
    test_df = test_df.copy()
    test_df["lgb_tmax_pred"] = lgb_model.predict(test_df[stage1_cols])

    X = transform_features(scaler, test_df, feature_cols)
    y = test_df[TARGET].to_numpy(dtype=float)
    mu, _sigma_raw, _ = predict_dist_params(model, X)

    out = test_df[["city", "date"]].copy()
    out["y_actual"] = y
    out["mu"] = mu
    return out


def compute_hit_rate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add continuous vs rounded hit-rate columns."""
    mu = frame["mu"].to_numpy(dtype=float)
    y = frame["y_actual"].to_numpy(dtype=float)

    continuous_hit = np.abs(mu - y) <= 1.0
    rounded_hit = np.abs(np.round(mu) - y) <= 1.0

    cont_lo = np.array([bucket_lower_bound(m) for m in mu])
    round_lo = np.array([bucket_lower_bound(r) for r in np.round(mu)])
    actual_lo = np.array([bucket_lower_bound(a) for a in y])

    continuous_modal = cont_lo == actual_lo
    rounded_modal = round_lo == actual_lo

    out = frame.copy()
    out["continuous_hit"] = continuous_hit
    out["rounded_hit"] = rounded_hit
    out["continuous_modal"] = continuous_modal
    out["rounded_modal"] = rounded_modal
    return out


def summarize_test1(frame: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for city, grp in frame.groupby("city"):
        n = len(grp)
        cont_1f = float(grp["continuous_hit"].mean())
        round_1f = float(grp["rounded_hit"].mean())
        cont_modal = float(grp["continuous_modal"].mean())
        round_modal = float(grp["rounded_modal"].mean())
        rows.append(
            {
                "city": str(city),
                "n": int(n),
                "continuous_1f_pct": round(100.0 * cont_1f, 2),
                "rounded_1f_pct": round(100.0 * round_1f, 2),
                "delta_1f_pp": round(100.0 * (round_1f - cont_1f), 2),
                "continuous_modal_pct": round(100.0 * cont_modal, 2),
                "rounded_modal_pct": round(100.0 * round_modal, 2),
                "delta_modal_pp": round(100.0 * (round_modal - cont_modal), 2),
            }
        )

    n_total = len(frame)
    overall = {
        "city": "OVERALL",
        "n": n_total,
        "continuous_1f_pct": round(100.0 * frame["continuous_hit"].mean(), 2),
        "rounded_1f_pct": round(100.0 * frame["rounded_hit"].mean(), 2),
        "delta_1f_pp": round(
            100.0 * (frame["rounded_hit"].mean() - frame["continuous_hit"].mean()), 2
        ),
        "continuous_modal_pct": round(100.0 * frame["continuous_modal"].mean(), 2),
        "rounded_modal_pct": round(100.0 * frame["rounded_modal"].mean(), 2),
        "delta_modal_pp": round(
            100.0 * (frame["rounded_modal"].mean() - frame["continuous_modal"].mean()), 2
        ),
    }
    rows.append(overall)

    delta_1f = overall["delta_1f_pp"]
    rounded_1f = overall["rounded_1f_pct"]
    cont_1f = overall["continuous_1f_pct"]
    delta_modal = overall["delta_modal_pp"]

    if rounded_1f >= 50.0 and delta_1f >= 10.0:
        verdict = (
            "Hypothesis 1 is **largely confirmed** for the ±1°F metric. Continuous scoring "
            f"({cont_1f:.1f}%) understates accuracy vs TrackB's rounded convention; rounded ±1°F "
            f"({rounded_1f:.1f}%) sits inside TrackB's 50–67% band (+{delta_1f:.1f}pp). "
            f"Modal bucket HR moves only +{delta_modal:.1f}pp with rounding "
            f"({overall['continuous_modal_pct']:.1f}% → {overall['rounded_modal_pct']:.1f}%), "
            "so any remaining TrackB gap on bucket hit rate is real, not a scoring artifact."
        )
    elif delta_1f >= 5.0:
        verdict = (
            "Rounding explains **some** of the ±1°F gap ({:.1f}pp), but NGBoost remains "
            "materially below TrackB even on a rounded basis ({}% vs 50–67%).".format(
                delta_1f, rounded_1f
            )
        )
    else:
        verdict = (
            "The TrackB gap is **real, not a measurement artifact**. Rounding mu before "
            f"scoring adds only {delta_1f:.1f}pp to ±1°F hit rate "
            f"({cont_1f:.1f}% → {rounded_1f:.1f}%), "
            "nowhere near TrackB's 50–67%."
        )

    return {"per_city": rows, "overall": overall, "verdict": verdict}


def iter_eligible_city_dates() -> list[tuple[str, str]]:
    """City-dates with entry-window snapshots and reliable WU settlement."""
    wu = load_wu_targets()
    pairs: list[tuple[str, str]] = []
    for city in POLY_CITIES:
        city_dir = PROJECT_ROOT / "data" / "polymarket_history" / "snapshots" / city
        if not city_dir.exists():
            continue
        for path in sorted(city_dir.glob("*.parquet")):
            date_str = path.stem
            frame = load_day_snapshot(city, date_str)
            if frame is None or not has_entry_window_snapshot(frame, city, date_str):
                continue
            wu_row = wu[(wu["city"] == city) & (wu["date"] == date_str)]
            if wu_row.empty or not bool(wu_row.iloc[0]["reliable"]):
                continue
            if not pd.notna(wu_row.iloc[0]["wunderground_tmax"]):
                continue
            pairs.append((city, date_str))
    return pairs


def settlement_bucket_index(actual_tmax: float) -> int:
    return bucket_lower_bound(actual_tmax)


def analyze_entry_snapshot(
    city: str,
    date_str: str,
    wu_actual: float,
) -> dict[str, Any] | None:
    frame = load_day_snapshot(city, date_str)
    if frame is None:
        return None
    snap_rows, _entry_ts, _excluded = select_entry_snapshot(frame, city, date_str)
    if snap_rows.empty:
        return None

    quotes = quotes_at_entry(snap_rows)
    if quotes.empty:
        return None

    quotes = quotes.copy()
    quotes["sort_key"] = quotes["bucket"].map(bucket_index_from_label)
    quotes = quotes.dropna(subset=["sort_key"])
    if quotes.empty:
        return None

    quotes["temp_mid"] = quotes["bucket"].map(bucket_midpoint_for_implied_mean)
    quotes = quotes.dropna(subset=["temp_mid"])
    if quotes.empty:
        return None

    prices = quotes["midpoint"].astype(float)
    if prices.sum() <= 0:
        return None
    implied_mean = float((quotes["temp_mid"] * prices).sum() / prices.sum())

    fav_idx = prices.idxmax()
    fav_row = quotes.loc[fav_idx]
    fav_label = str(fav_row["bucket"])
    fav_sort = int(fav_row["sort_key"])
    fav_price = float(fav_row["midpoint"])

    sorted_quotes = quotes.sort_values("sort_key").reset_index(drop=True)
    sort_keys = sorted_quotes["sort_key"].astype(int).tolist()
    fav_pos = sort_keys.index(fav_sort)

    b_minus_1_label: str | None = None
    b_minus_1_price: float | None = None
    b_plus_1_label: str | None = None
    if fav_pos > 0:
        b_minus_1_label = str(sorted_quotes.iloc[fav_pos - 1]["bucket"])
        b_minus_1_price = float(sorted_quotes.iloc[fav_pos - 1]["midpoint"])
    if fav_pos < len(sorted_quotes) - 1:
        b_plus_1_label = str(sorted_quotes.iloc[fav_pos + 1]["bucket"])

    actual_lo = settlement_bucket_index(wu_actual)
    fav_lo = fav_lo_from_label(fav_label)
    if fav_lo is None:
        return None

    delta_buckets = (actual_lo - fav_lo) // 2

    return {
        "city": city,
        "date": date_str,
        "wu_actual": float(wu_actual),
        "implied_mean": implied_mean,
        "bias": implied_mean - float(wu_actual),
        "favorite_label": fav_label,
        "favorite_sort_key": fav_lo,
        "favorite_price": fav_price,
        "b_minus_1_label": b_minus_1_label,
        "b_minus_1_price": b_minus_1_price,
        "b_plus_1_label": b_plus_1_label,
        "actual_bucket_lo": actual_lo,
        "delta_buckets": int(delta_buckets),
        "b_minus_1_wins": b_minus_1_label is not None and actual_lo == bucket_index_from_label(b_minus_1_label),
    }


def fav_lo_from_label(label: str) -> int | None:
    import re

    text = str(label).strip()
    rng = re.match(r"^(\d+)-(\d+)$", text)
    if rng:
        return int(rng.group(1))
    le = re.match(r"^<=(\d+)$", text)
    if le:
        return bucket_lower_bound(float(int(le.group(1))))
    ge = re.match(r"^>=(\d+)$", text)
    if ge:
        return int(ge.group(1))
    return None


def describe_bias_histogram(bias: np.ndarray) -> str:
    if len(bias) == 0:
        return "No data."
    hist, edges = np.histogram(bias, bins=20)
    peak_bin = int(np.argmax(hist))
    peak_center = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
    mean_b = float(np.mean(bias))
    median_b = float(np.median(bias))
    pct_pos = 100.0 * float(np.mean(bias > 0))
    skew = "right-skewed" if mean_b > median_b + 0.05 else (
        "left-skewed" if mean_b < median_b - 0.05 else "roughly symmetric"
    )
    return (
        f"Distribution of bias (implied_mean − WU actual) across {len(bias)} city-dates: "
        f"mean={mean_b:.2f}°F, median={median_b:.2f}°F, {pct_pos:.1f}% positive. "
        f"Histogram peaks near {peak_center:.1f}°F (bin count {hist[peak_bin]}). "
        f"Shape is {skew}."
    )


def run_test2() -> dict[str, Any]:
    wu = load_wu_targets()
    rows: list[dict[str, Any]] = []
    for city, date_str in iter_eligible_city_dates():
        wu_row = wu[(wu["city"] == city) & (wu["date"] == date_str)]
        wu_actual = float(wu_row.iloc[0]["wunderground_tmax"])
        rec = analyze_entry_snapshot(city, date_str, wu_actual)
        if rec is not None:
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0, "verdict": "No eligible city-dates with entry snapshots.", "rows": []}

    bias = df["bias"].to_numpy(dtype=float)
    t_stat, p_value = stats.ttest_1samp(bias, popmean=0.0, alternative="greater")

    overall = {
        "n": int(len(df)),
        "mean_bias": round(float(np.mean(bias)), 3),
        "median_bias": round(float(np.median(bias)), 3),
        "std_bias": round(float(np.std(bias, ddof=1)), 3),
        "pct_positive": round(100.0 * float(np.mean(bias > 0)), 2),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 6),
        "histogram_description": describe_bias_histogram(bias),
    }

    per_city: list[dict[str, Any]] = []
    for city, grp in df.groupby("city"):
        b = grp["bias"].to_numpy(dtype=float)
        ts, pv = stats.ttest_1samp(b, popmean=0.0, alternative="greater")
        per_city.append(
            {
                "city": str(city),
                "n": int(len(grp)),
                "mean_bias": round(float(np.mean(b)), 3),
                "median_bias": round(float(np.median(b)), 3),
                "pct_positive": round(100.0 * float(np.mean(b > 0)), 2),
                "t_stat": round(float(ts), 3),
                "p_value": round(float(pv), 6),
            }
        )

    delta = df["delta_buckets"].astype(int)
    settlement_table = {
        "lands_in_favorite_B": int((delta == 0).sum()),
        "lands_in_B_minus_1": int((delta == -1).sum()),
        "lands_in_B_plus_1": int((delta == 1).sum()),
        "lands_in_B_minus_2_or_lower": int((delta <= -2).sum()),
        "lands_in_B_plus_2_or_higher": int((delta >= 2).sum()),
    }
    n_settle = len(df)
    settlement_pct = {
        k: round(100.0 * v / n_settle, 2) for k, v in settlement_table.items()
    }

    b_minus_prices = df.dropna(subset=["b_minus_1_price"])
    b_minus_win_rate = settlement_table["lands_in_B_minus_1"] / n_settle
    avg_b_minus_price = (
        float(b_minus_prices["b_minus_1_price"].mean()) if len(b_minus_prices) else None
    )
    expected_pnl = None
    if avg_b_minus_price is not None:
        wr = b_minus_win_rate
        expected_pnl = wr * (1.0 - avg_b_minus_price) - (1.0 - wr) * avg_b_minus_price

    significant = p_value < 0.05
    if significant and overall["mean_bias"] > 0:
        if settlement_table["lands_in_B_minus_1"] > settlement_table["lands_in_B_plus_1"]:
            edge_note = (
                f"Significant positive bias ({overall['mean_bias']:.2f}°F, p={p_value:.4f}). "
                f"Settlement lands in B−1 ({settlement_pct['lands_in_B_minus_1']:.1f}%) more often "
                f"than B+1 ({settlement_pct['lands_in_B_plus_1']:.1f}%), consistent with anchoring."
            )
        else:
            edge_note = (
                f"Significant positive bias ({overall['mean_bias']:.2f}°F, p={p_value:.4f}), "
                "but settlement does not systematically land below the market favorite."
            )
    elif significant:
        edge_note = "Bias test significant but mean bias is not positive."
    else:
        edge_note = (
            f"No significant anchoring bias detected (mean={overall['mean_bias']:.2f}°F, p={p_value:.4f})."
        )

    pnl_verdict = "N/A — bias not significant at p<0.05."
    if significant and expected_pnl is not None:
        pnl_verdict = (
            f"Naive always-buy-B−1 at entry: avg price={avg_b_minus_price:.3f}, "
            f"win rate={100.0 * b_minus_win_rate:.1f}%, "
            f"expected PnL per $1 risked={expected_pnl:.3f} "
            f"({'positive' if expected_pnl > 0 else 'negative'} at maker/zero fee)."
        )

    return {
        "n": n_settle,
        "overall": overall,
        "per_city": per_city,
        "settlement_vs_favorite": settlement_table,
        "settlement_vs_favorite_pct": settlement_pct,
        "b_minus_1_strategy": {
            "avg_entry_price": round(avg_b_minus_price, 4) if avg_b_minus_price else None,
            "win_rate": round(b_minus_win_rate, 4),
            "expected_pnl_per_dollar": round(expected_pnl, 4) if expected_pnl is not None else None,
            "n_with_b_minus_1": int(len(b_minus_prices)),
        },
        "verdict": edge_note,
        "pnl_verdict": pnl_verdict,
        "rows": rows,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    t1 = payload["test1"]
    t2 = payload["test2"]

    lines = [
        "# Fable Diagnostics\n\n",
        "Two tests to evaluate whether the NGBoost–TrackB gap is a measurement artifact "
        "and whether Polymarket prices are anchored above WU settlement.\n\n",
        "## Test 1: Hit Rate Recompute\n\n",
        "NGBoost v2 on the 2026 test set. Compares continuous μ scoring vs rounding μ "
        "before ±1°F and modal-bucket checks (2°F Polymarket buckets).\n\n",
        "| City | N | Continuous ±1F | Rounded ±1F | Delta | Cont Modal HR | Round Modal HR | Delta |\n",
        "|------|--:|---------------:|------------:|------:|----------------:|---------------:|------:|\n",
    ]
    for row in t1["per_city"]:
        lines.append(
            f"| {row['city']} | {row['n']} | {row['continuous_1f_pct']:.1f}% | "
            f"{row['rounded_1f_pct']:.1f}% | {row['delta_1f_pp']:+.1f}pp | "
            f"{row['continuous_modal_pct']:.1f}% | {row['rounded_modal_pct']:.1f}% | "
            f"{row['delta_modal_pp']:+.1f}pp |\n"
        )

    o = t1["overall"]
    lines.extend(
        [
            "\n**Overall summary:**\n\n",
            f"- Continuous ±1°F: **{o['continuous_1f_pct']:.1f}%**\n",
            f"- Rounded ±1°F: **{o['rounded_1f_pct']:.1f}%** ({o['delta_1f_pp']:+.1f}pp vs continuous)\n",
            f"- Continuous modal bucket HR: **{o['continuous_modal_pct']:.1f}%**\n",
            f"- Rounded modal bucket HR: **{o['rounded_modal_pct']:.1f}%** "
            f"({o['delta_modal_pp']:+.1f}pp vs continuous)\n\n",
            f"**VERDICT:** {t1['verdict']}\n\n",
            "## Test 2: Settlement Anchoring\n\n",
            f"Entry-window snapshots for **{t2.get('n', 0)}** eligible city-dates "
            "(Telonex order book + WU settlement).\n\n",
            "### a) Overall bias statistics\n\n",
        ]
    )

    if t2.get("n", 0) == 0:
        lines.append("No data available.\n\n")
    else:
        ov = t2["overall"]
        lines.extend(
            [
                f"- Mean bias (implied_mean − WU actual): **{ov['mean_bias']:+.2f}°F**\n",
                f"- Median bias: **{ov['median_bias']:+.2f}°F**\n",
                f"- Std of bias: **{ov['std_bias']:.2f}°F**\n",
                f"- Fraction bias > 0 (market too high): **{ov['pct_positive']:.1f}%**\n",
                f"- One-sided t-test (H₀: mean bias ≤ 0): t={ov['t_stat']:.3f}, "
                f"p={ov['p_value']:.6f}\n\n",
                "### b) Per-city breakdown\n\n",
                "| City | N | Mean bias | Median bias | Pct positive | t-stat | p-value |\n",
                "|------|--:|----------:|------------:|-------------:|-------:|--------:|\n",
            ]
        )
        for row in t2["per_city"]:
            lines.append(
                f"| {row['city']} | {row['n']} | {row['mean_bias']:+.2f} | "
                f"{row['median_bias']:+.2f} | {row['pct_positive']:.1f}% | "
                f"{row['t_stat']:.3f} | {row['p_value']:.4f} |\n"
            )

        lines.extend(
            [
                "\n### c) Histogram of bias\n\n",
                f"{ov['histogram_description']}\n\n",
                "### d) Settlement vs market favorite\n\n",
                "| Settlement vs market favorite | Count | Pct |\n",
                "|-------------------------------|------:|----:|\n",
            ]
        )
        labels = [
            ("lands_in_favorite_B", "Lands in favorite (B)"),
            ("lands_in_B_minus_1", "Lands in B−1 (one below)"),
            ("lands_in_B_plus_1", "Lands in B+1 (one above)"),
            ("lands_in_B_minus_2_or_lower", "Lands in B−2 or lower"),
            ("lands_in_B_plus_2_or_higher", "Lands in B+2 or higher"),
        ]
        for key, label in labels:
            cnt = t2["settlement_vs_favorite"][key]
            pct = t2["settlement_vs_favorite_pct"][key]
            lines.append(f"| {label} | {cnt} | {pct:.1f}% |\n")

        strat = t2["b_minus_1_strategy"]
        lines.extend(
            [
                "\n### e) Naive B−1 strategy (if bias significant)\n\n",
            ]
        )
        if strat["avg_entry_price"] is not None:
            lines.extend(
                [
                    f"- Average B−1 entry price: **{strat['avg_entry_price']:.3f}**\n",
                    f"- B−1 win rate: **{100.0 * strat['win_rate']:.1f}%**\n",
                    f"- Expected PnL per $1 risked: **{strat['expected_pnl_per_dollar']:.3f}**\n",
                    f"- Maker fee assumption: zero\n\n",
                ]
            )
        lines.extend(
            [
                f"**VERDICT:** {t2['verdict']}\n\n",
                f"**PnL:** {t2['pnl_verdict']}\n\n",
            ]
        )

    lines.extend(
        [
            "## Combined Implications\n\n",
            payload["combined_implications"],
            "\n",
        ]
    )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("".join(lines), encoding="utf-8")


def combined_implications(t1: dict[str, Any], t2: dict[str, Any]) -> str:
    parts: list[str] = []

    o = t1["overall"]
    if o["rounded_1f_pct"] >= 50.0 and o["delta_1f_pp"] >= 10.0:
        parts.append(
            "Test 1: the ±1°F TrackB comparison was mostly a **phantom gap** — NGBoost v2 is "
            f"at {o['rounded_1f_pct']:.1f}% under TrackB's rounded scoring convention "
            f"(vs {o['continuous_1f_pct']:.1f}% continuous). Recent μ-tuning may have been "
            f"misdirected if the target metric was ±1°F. Modal bucket HR remains ~{o['rounded_modal_pct']:.0f}% "
            f"(+{o['delta_modal_pp']:.1f}pp from rounding only); confirm which metric TrackB reports."
        )
    elif o["delta_1f_pp"] < 5.0:
        parts.append(
            "Test 1 shows rounding does **not** explain the NGBoost–TrackB gap: even with "
            f"TrackB-style rounded scoring, ±1°F hit rate is only {o['rounded_1f_pct']:.1f}% "
            f"vs TrackB's 50–67%. Continued work on μ accuracy is warranted, not abandoned."
        )
    elif o["delta_1f_pp"] >= 5.0:
        parts.append(
            "Test 1 shows rounding closes part of the ±1°F gap "
            f"({o['continuous_1f_pct']:.1f}% → {o['rounded_1f_pct']:.1f}%). "
            "Reconcile TrackB evaluation methodology before prioritizing further μ tuning."
        )

    if t2.get("n", 0) > 0:
        ov = t2["overall"]
        st = t2["settlement_vs_favorite"]
        if ov["p_value"] < 0.05 and ov["mean_bias"] > 0:
            if st["lands_in_B_minus_1"] > st["lands_in_B_plus_1"]:
                parts.append(
                    "Test 2 supports a **structural anchoring edge**: the market-implied mean "
                    f"is {ov['mean_bias']:+.2f}°F above WU settlement (p={ov['p_value']:.4f}), "
                    "and actuals land one bucket below the favorite more often than one above. "
                    "A contrarian B−1 maker strategy may capture durable edge independent of model μ."
                )
            else:
                parts.append(
                    "Test 2 finds significant upward bias in market-implied temperature, "
                    "but settlement does not consistently land below the favorite — anchoring "
                    "may inflate prices without a simple one-bucket contrarian trade."
                )
        else:
            parts.append(
                "Test 2 does **not** confirm systematic NWS-style anchoring above WU settlement "
                f"(mean bias {ov['mean_bias']:+.2f}°F, p={ov['p_value']:.4f}). "
                "Market pricing appears aligned with WU settlement on average."
            )

    if o["delta_1f_pp"] < 5.0 and t2.get("n", 0) > 0:
        ov = t2["overall"]
        if ov["p_value"] < 0.05 and t2["settlement_vs_favorite"]["lands_in_B_minus_1"] > t2[
            "settlement_vs_favorite"
        ]["lands_in_B_plus_1"]:
            parts.append(
                "**Strategic direction:** Pursue dual-track — keep improving NGBoost μ for "
                "model-based edges, while exploiting structural B−1 underpricing from forecast anchoring."
            )
        else:
            parts.append(
                "**Strategic direction:** Focus on closing the real μ accuracy gap (Test 1); "
                "do not rely on anchoring contrarian trades without stronger Test 2 confirmation."
            )

    return "\n\n".join(parts) + "\n"


def main() -> None:
    print("Test 1: loading NGBoost v2 test predictions...")
    pred_df = load_test_predictions()
    pred_df = compute_hit_rate_metrics(pred_df)
    test1 = summarize_test1(pred_df)
    print(f"  {test1['overall']['n']} test rows, continuous ±1F={test1['overall']['continuous_1f_pct']:.1f}%")

    print("Test 2: scanning Telonex entry snapshots...")
    test2 = run_test2()
    print(f"  {test2.get('n', 0)} eligible city-dates")

    payload: dict[str, Any] = {
        "model_dir": str(MODEL_DIR),
        "test1": test1,
        "test2": {k: v for k, v in test2.items() if k != "rows"},
        "combined_implications": combined_implications(test1, test2),
        "raw": {
            "test1_predictions": pred_df.to_dict(orient="records"),
            "test2_city_dates": test2.get("rows", []),
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    write_markdown(payload)
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
