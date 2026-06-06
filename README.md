# MCP Project

This repository contains a small research/backtesting workspace for Kalshi daily high-temperature (`Tmax`) market data. It loads raw city-level CSV exports, builds train/test parquet partitions, and provides a notebook-based interactive explorer for inspecting how bucket probabilities evolve through each trading day.

## Folder Structure

```text
.
|-- data/
|   `-- splits/
|       |-- threshold_opt.parquet
|       |-- time_holdout.parquet
|       |-- location_holdout.parquet
|       |-- true_holdout.parquet
|       |-- frozen_k.json
|       `-- smoke_test_results/
|-- historic_tmax_market_data/
|   |-- austin/
|   |-- chicago_midway/
|   |-- denver/
|   |-- houston/
|   |-- los_angeles/
|   |-- miami/
|   |-- minneapolis/
|   |-- new_york_city/
|   |-- oklahoma_city/
|   |-- philadelphia/
|   |-- phoenix/
|   `-- san_francisco/
|-- notebooks/
|   `-- market_explorer.ipynb
|-- scripts/
|   |-- build_splits.py
|   `-- run_baselines_smoke_test.py
`-- src/
    |-- entry_interface.py
    |-- fees.py
    |-- market_explorer_widgets.py
    |-- snapshot_stability.py
    |-- backtest_utils.py
    `-- baselines/
        |-- __init__.py
        |-- implied_favorite.py
        |-- implied_distribution_copy.py
        `-- sell_longshots.py
```

## Data Folders

### `historic_tmax_market_data/`

Raw city-level Kalshi/NWS Tmax market exports live here. Each city has its own folder containing a CSV file plus a small schema/readme file. The notebook and split builder discover CSVs with this pattern:

```text
*/*tmax_kalshi*5min_same_day.csv
```

Current city folders:

- `austin/`
- `chicago_midway/`
- `denver/`
- `houston/`
- `los_angeles/`
- `miami/`
- `minneapolis/`
- `new_york_city/`
- `oklahoma_city/`
- `philadelphia/`
- `phoenix/`
- `san_francisco/`

The raw rows represent bucket-level market snapshots for one city/date. Important columns include `event_date`, `city`, `snapshot_time_local`, `bucket_label`, `yes_mid_close`, settlement/resolution columns, volume/open interest, and NWS Tmax metadata. Some exports, such as Austin, may not include a `city` column; project code fills it from the folder name.

### `data/splits/`

Generated parquet partitions built from the raw CSVs by `scripts/build_splits.py`.

- `threshold_opt.parquet`: training partition used to tune strategy parameters such as the snapshot-stability `k`.
- `time_holdout.parquet`: date holdout partition for cities included in the training city set.
- `location_holdout.parquet`: city holdout partition for cities outside the training city set.
- `true_holdout.parquet`: final holdout subset intended to remain untouched until final reporting.
- `frozen_k.json`: persisted snapshot-stability `k` (`{"k": <int>}`), created on first run by `src/snapshot_stability.py` and reused by all baselines so `k` is never re-optimized inside a baseline.
- `smoke_test_results/`: in-sample baseline result parquets written by `scripts/run_baselines_smoke_test.py` (`implied_favorite_IS.parquet`, `distribution_copy_IS.parquet`, `sell_longshots_IS.parquet`).

With the current 12-city dataset, all cities fall into the training city set because `TRAIN_CITY_COUNT = 20`, so the location/true holdout files are currently empty.

Regenerate these files with:

```bash
python scripts/build_splits.py
```

The script requires a pandas parquet engine such as `pyarrow` or `fastparquet`.

## Notebooks

### `notebooks/market_explorer.ipynb`

Interactive exploratory notebook for the raw market data.

The first cell loads all city CSVs into `market_df`, normalizes date/timestamp columns, fills missing city labels from folder names when needed, and prints the number of rows and city CSVs loaded.

The second cell displays an interactive widget from `src/market_explorer_widgets.py`. It lets you choose a city and event date, then slide through intraday snapshots to see the market-implied probability distribution across temperature buckets. The chart highlights the modal bucket and outlines the resolved winning bucket.

## Scripts

### `scripts/build_splits.py`

Builds parquet split files from `historic_tmax_market_data/`.

Main behavior:

- Discovers one raw CSV per city folder.
- Normalizes `event_date`, `snapshot_time_local`, `source_city_folder`, and missing `city` labels.
- Sorts cities alphabetically.
- Uses the first `TRAIN_CITY_COUNT` cities as train cities.
- Splits train-city dates into `threshold_opt` and `time_holdout`.
- Uses remaining cities, when present, for `location_holdout` and `true_holdout`.
- Writes the four parquet files under `data/splits/`.
- Prints train/holdout city lists and partition summaries.

### `scripts/run_baselines_smoke_test.py`

Runs all three baselines in-sample on `threshold_opt.parquet` to verify the code executes correctly.

Main behavior:

- Loads `threshold_opt.parquet` and the frozen `k` via `load_or_create_frozen_k()`.
- Evaluates `implied_favorite`, `implied_distribution_copy`, and `sell_longshots`.
- Prints a `print_summary_table` block for each baseline.
- Saves the three result dataframes to `data/splits/smoke_test_results/`.
- Prints a reminder that these results are in-sample and not OOS performance.

Run it with:

```bash
python scripts/run_baselines_smoke_test.py
```

Key constants:

- `EXPECTED_CITY_COUNT = 27`
- `TRAIN_CITY_COUNT = 20`
- `EXPECTED_TEST_DAYS = 48`
- `THRESHOLD_OPT_DAYS = 33`
- `TIME_HOLDOUT_DAYS = 15`

## Source Modules

### `src/market_explorer_widgets.py`

Contains the notebook widget implementation for the market explorer.

Key function:

- `display_market_explorer(market_df, frozen_k=2)`: displays city/date dropdowns, a snapshot slider, the evolving probability histogram, and summary metrics.

The helper also contains internal utilities for bucket sorting, Shannon entropy, snapshot extraction, and checking whether the modal bucket has stabilized for `k` consecutive snapshots.

### `src/entry_interface.py`

Defines the shared interface for point-in-time entry rules.

Key pieces:

- `TradeSignal`: dataclass describing one trade-entry decision.
- `make_entry_rule`: decorator that enforces the entry-rule contract and guards against selecting a snapshot time that is not present in the provided day dataframe.

### `src/fees.py`

Kalshi fee and PnL utilities for backtests.

Key functions:

- `taker_fee(C, P)`: taker fee in cents.
- `maker_fee(C, P)`: maker fee in cents.
- `net_pnl(gross_pnl_cents, C, P, order_type="taker")`: PnL after fees.

Running this file directly prints a small fee sanity-check table.

### `src/snapshot_stability.py`

Implements and optimizes the snapshot-stability entry rule.

Key functions:

- `compute_modal_bucket(day_df, snapshot_time)`: returns the bucket with the highest `yes_mid_close` at a snapshot.
- `stability_entry(day_df, k)`: emits a `TradeSignal` when the modal bucket has stayed unchanged for `k` consecutive snapshots.
- `optimise_k(partition_df, k_grid=None)`: evaluates candidate `k` values on a partition and chooses the best finite Sharpe ratio after fees.
- `load_or_create_frozen_k(split_dir=SPLIT_DIR)`: the single source of truth for the frozen `k`. Loads `data/splits/frozen_k.json` if present, otherwise runs `optimise_k` on `threshold_opt.parquet`, persists the result, and returns it.

Running this file directly persists the frozen `k` to `data/splits/frozen_k.json` and prints the value to use in later baselines.

### `src/backtest_utils.py`

Reusable backtest evaluation helpers shared by every baseline.

Key functions:

- `daily_returns(results_df, pnl_col="net_pnl_cents", capital=100.0)`: converts a per-trade results frame into a daily return series (no-signal days contribute 0).
- `sharpe_stats(returns)`: returns a dict of summary statistics including annualised Sharpe, its Lo (2002) standard error and 95% CI, max drawdown, annualised Sortino, and lag-1 autocorrelation.
- `print_summary_table(baseline_name, results_df, capital=100.0)`: prints a clean one-block performance summary for a baseline result frame.
- `bucket_sort_key(bucket_label)`: numeric bucket sort key (LESS_THAN, then RANGE, then GREATER_THAN) for display ordering.

### `src/baselines/`

Baseline trading strategies, each exposing point-in-time entry rules plus an `evaluate_*` driver and a self-contained `__main__` smoke test:

- `implied_favorite.py`: buys YES on the modal bucket at the snapshot-stability entry.
- `implied_distribution_copy.py`: at the same entry snapshot, bets every bucket in proportion to its fee-adjusted market price (the no-edge benchmark).
- `sell_longshots.py`: sells YES (buys NO) on buckets that first cross below a low YES price after the stability window, fading the favorite-longshot bias.

## Typical Workflow

1. Add or update city CSV exports under `historic_tmax_market_data/<city>/`.
2. Rebuild parquet splits:

   ```bash
   python scripts/build_splits.py
   ```

3. Open `notebooks/market_explorer.ipynb`.
4. Run the first cell to load `market_df`.
5. Run the second cell to launch the interactive market explorer.
6. Use `src/snapshot_stability.py` to optimize the snapshot-stability entry rule on `threshold_opt.parquet` and persist the frozen `k`.
7. Run `python scripts/run_baselines_smoke_test.py` to evaluate the baselines in-sample and write results to `data/splits/smoke_test_results/`.

## Notes

- Raw CSV files and generated parquet files can be large and may be hidden from normal workspace search by ignore rules.
- The notebook currently reports `Loaded 265,752 rows from 12 city CSVs` with the current local data.
- The project assumes local execution from the repository root or from the `notebooks/` directory.
