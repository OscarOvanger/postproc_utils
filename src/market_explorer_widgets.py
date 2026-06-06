"""Interactive widgets for exploring Kalshi Tmax market snapshots."""

from __future__ import annotations

import math

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display


def _bucket_sort_key(row: pd.Series) -> tuple[int, float | str]:
    """Sort bucket rows by temperature order, not label lexicographic order."""
    bucket_type = row.get("bucket_type")
    lower = row.get("bucket_lower_inclusive_f")
    upper = row.get("bucket_upper_inclusive_f")
    lower = float(lower) if pd.notna(lower) else math.inf
    upper = float(upper) if pd.notna(upper) else math.inf
    if bucket_type == "LESS_THAN":
        return (0, upper)
    if bucket_type == "RANGE":
        return (1, lower)
    if bucket_type == "GREATER_THAN":
        return (2, lower)
    return (3, str(row.get("bucket_label")))


def _shannon_entropy(probabilities: np.ndarray) -> float:
    """Return Shannon entropy for a probability vector."""
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[np.isfinite(probs) & (probs > 0)]
    return float(-(probs * np.log2(probs)).sum())


def _snapshot_rows_for_index(
    day_df: pd.DataFrame, snapshot_index: int
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Return rows at the selected snapshot index."""
    times = list(day_df["snapshot_time_local"].drop_duplicates().sort_values())
    if not times:
        return day_df.iloc[0:0], None
    snapshot_time = times[min(snapshot_index, len(times) - 1)]
    rows = day_df[day_df["snapshot_time_local"] == snapshot_time].copy()
    rows["_bucket_sort_key"] = rows.apply(_bucket_sort_key, axis=1)
    rows = rows.sort_values("_bucket_sort_key")
    return rows, snapshot_time


def _stability_triggered_by(
    day_df: pd.DataFrame, snapshot_time: pd.Timestamp, k: int
) -> bool:
    """Return whether the modal bucket has stabilized by the given snapshot."""
    elapsed = day_df[day_df["snapshot_time_local"] <= snapshot_time]
    modes: list[str] = []
    for time in elapsed["snapshot_time_local"].drop_duplicates().sort_values():
        rows = elapsed[elapsed["snapshot_time_local"] == time]
        best = rows.loc[rows["yes_mid_close"].astype(float).idxmax()]
        modes.append(str(best["bucket_label"]))
        if len(modes) >= k and len(set(modes[-k:])) == 1:
            return True
    return False


def _plot_snapshot(
    rows: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    city: str,
    event_date: str,
) -> None:
    """Render the bucket probability chart for one snapshot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = rows["bucket_label"].astype(str).tolist()
    probabilities = rows["yes_mid_close"].astype(float).to_numpy()
    modal_idx = int(np.nanargmax(probabilities))
    winners = rows["bucket_resolved_to_one_dollars"].astype(bool).to_numpy()
    colors = ["tab:blue" if idx == modal_idx else "lightgray" for idx in range(len(rows))]

    bars = ax.bar(labels, probabilities, color=colors, edgecolor="gray", linewidth=1)
    for bar, is_winner in zip(bars, winners, strict=False):
        if is_winner:
            bar.set_edgecolor("tab:red")
            bar.set_linewidth(3)

    entropy = _shannon_entropy(probabilities)
    minutes_before_tmax = rows["minutes_before_tmax"].iloc[0]
    title_time = pd.Timestamp(snapshot_time).strftime("%H:%M")
    ax.set_ylim(0, 1)
    ax.set_ylabel("yes_mid_close")
    ax.set_title(
        f"{city} | {event_date} | {title_time} | minutes_before_tmax={minutes_before_tmax}"
    )
    ax.text(
        0.98,
        0.95,
        f"H = {entropy:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plt.show()


def display_market_explorer(market_df: pd.DataFrame, frozen_k: int = 2) -> widgets.VBox:
    """Display the interactive market snapshot explorer for a loaded market dataframe."""
    city_dropdown = widgets.Dropdown(
        options=sorted(market_df["city"].dropna().unique()),
        description="city:",
        layout=widgets.Layout(width="320px"),
    )
    event_date_dropdown = widgets.Dropdown(
        description="event_date:",
        layout=widgets.Layout(width="260px"),
    )
    snapshot_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="snapshot:",
        continuous_update=True,
        layout=widgets.Layout(width="520px"),
    )
    snapshot_label = widgets.Label()
    chart_output = widgets.Output()
    summary_output = widgets.Output()

    def event_dates_for_city(city: str) -> list[str]:
        city_rows = market_df[market_df["city"] == city]
        return sorted(city_rows["event_date"].dropna().unique())

    def current_day_df() -> pd.DataFrame:
        mask = (
            (market_df["city"] == city_dropdown.value)
            & (market_df["event_date"] == event_date_dropdown.value)
        )
        return market_df.loc[mask].sort_values(["snapshot_time_local", "bucket_label"])

    def current_snapshot_times() -> list[pd.Timestamp]:
        day_df = current_day_df()
        return list(day_df["snapshot_time_local"].drop_duplicates().sort_values())

    def refresh_event_dates(*_: object) -> None:
        dates = event_dates_for_city(city_dropdown.value)
        event_date_dropdown.options = dates
        if dates:
            event_date_dropdown.value = dates[0]

    def refresh_snapshot_label(*_: object) -> None:
        times = current_snapshot_times()
        if not times:
            snapshot_label.value = "no snapshots"
            return
        snapshot_time = times[snapshot_slider.value]
        snapshot_label.value = pd.Timestamp(snapshot_time).strftime("%H:%M")

    def refresh_snapshot_slider(*_: object) -> None:
        times = current_snapshot_times()
        snapshot_slider.max = max(len(times) - 1, 0)
        snapshot_slider.value = min(snapshot_slider.value, snapshot_slider.max)
        refresh_snapshot_label()

    def update_outputs(*_: object) -> None:
        day_df = current_day_df()
        rows, snapshot_time = _snapshot_rows_for_index(day_df, snapshot_slider.value)
        if rows.empty or snapshot_time is None:
            with chart_output:
                clear_output(wait=True)
                print("No snapshot rows available")
            with summary_output:
                clear_output(wait=True)
            return

        probabilities = rows["yes_mid_close"].astype(float).to_numpy()
        modal_idx = int(np.nanargmax(probabilities))
        modal_row = rows.iloc[modal_idx]
        mode_prob = float(modal_row["yes_mid_close"])
        entropy = _shannon_entropy(probabilities)

        times = list(day_df["snapshot_time_local"].drop_duplicates().sort_values())
        previous_idx = max(snapshot_slider.value - 5, 0)
        previous_time = times[previous_idx]
        previous_rows = day_df[day_df["snapshot_time_local"] == previous_time]
        previous_mode_prob = float(previous_rows["yes_mid_close"].astype(float).max())
        delta_mode = mode_prob - previous_mode_prob
        triggered = _stability_triggered_by(day_df, snapshot_time, frozen_k)

        with chart_output:
            clear_output(wait=True)
            _plot_snapshot(rows, snapshot_time, city_dropdown.value, event_date_dropdown.value)

        with summary_output:
            clear_output(wait=True)
            summary = widgets.HTML(
                "<div style='font-family: monospace; line-height: 1.6'>"
                f"mode_prob: {mode_prob:.4f}<br>"
                f"entropy H: {entropy:.4f}<br>"
                f"delta_mode vs 5 snapshots ago: {delta_mode:+.4f}<br>"
                f"stability rule (k={frozen_k}) triggered yet: {triggered}"
                "</div>"
            )
            display(summary)

    city_dropdown.observe(refresh_event_dates, names="value")
    city_dropdown.observe(refresh_snapshot_slider, names="value")
    event_date_dropdown.observe(refresh_snapshot_slider, names="value")
    snapshot_slider.observe(refresh_snapshot_label, names="value")
    for widget in [city_dropdown, event_date_dropdown, snapshot_slider]:
        widget.observe(update_outputs, names="value")

    refresh_event_dates()
    refresh_snapshot_slider()
    update_outputs()

    explorer = widgets.VBox(
        [
            widgets.HBox([city_dropdown, event_date_dropdown]),
            widgets.HBox([snapshot_slider, snapshot_label]),
            chart_output,
            summary_output,
        ]
    )
    display(explorer)
    return explorer
