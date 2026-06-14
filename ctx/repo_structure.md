# Repository Structure

## Overview

This repo contains the trading strategy code for a statistical post-processing pipeline. Sensitive configuration, model artifacts, and data files are gitignored and stored locally or on HuggingFace.

## Source Code (`src/`)

### Core Modules

| File | Description |
|------|-------------|
| `data_pipeline.py` | Unified Track-B feature construction for historical backfill and live daily fetch; orchestrates ASOS, lags, NWS MOS, GFS, and NWP sources with leakage assertions. |
| `data_store.py` | Unified local / HuggingFace data access for Track-B features and train/test splits. |
| `backtest_utils.py` | Shared backtest evaluation helpers reused across all baseline strategies (Sharpe, drawdown, summary tables). |
| `entry_interface.py` | Entry-rule interface and look-ahead guard for baseline strategies (`TradeSignal`, `make_entry_rule`). |
| `fees.py` | Exchange fee utilities for backtesting Tmax bucket trades (taker/maker fees, net PnL). |
| `sizing.py` | Kelly sizing helpers, edge guardrails (`has_edge`: model_prob − price > 2× fee), and contract sizing utilities. |
| `snapshot_stability.py` | Snapshot-stability entry rule and optimization utilities; loads or creates frozen `k` in `data/splits/frozen_k.json`. |
| `frozen_params.py` | Load and persist frozen threshold parameters to `data/splits/frozen_params.json`. |
| `optimisation_harness.py` | Walk-forward optimisation harness for threshold baselines. |
| `market_explorer_widgets.py` | Interactive widgets for exploring Tmax market snapshots (city/date dropdowns, intraday probability charts). |
| `kalshi_api.py` | Market exchange API wrapper; paper mode by default (live order placement stub). |
| `risk_monitor.py` | Risk monitoring for the MCP 60-day challenge (bankroll, drawdown, trade pace). |

### Baselines (`src/baselines/`)

| File | Description |
|------|-------------|
| `implied_favorite.py` | Implied-favorite baseline: buy YES on the stabilized modal bucket. |
| `implied_distribution_copy.py` | Implied-distribution-copy baseline: bet every bucket in market proportion. |
| `sell_longshots.py` | Sell-longshots baseline: fade buckets that cross below a low YES price. |
| `make_the_market.py` | Make-the-market baseline: quote one cent inside the spread on the modal bucket. |
| `mode_prob_threshold.py` | Mode-probability threshold baseline: enter when mode_prob ≥ t_star. |
| `entropy_threshold.py` | Entropy threshold baseline: enter when distribution entropy ≤ h_star. |
| `momentum_threshold.py` | Momentum threshold baseline: enter when delta_mode ≥ d_star. |
| `track_j_flat.py` | Track-J flat baseline: buy YES on the highest Track-J probability bucket. |
| `__init__.py` | Baseline trading strategies for the Tmax backtesting workspace. |

### Models (`src/models/`)

| File | Description |
|------|-------------|
| `track_j.py` | Track-J forecast interface and bucket probability conversion (Gaussian CDF over 2°F buckets). |
| `__init__.py` | Model interfaces used by market entry rules. |

### Feature Engineering (`src/trackj/`)

| File | Description |
|------|-------------|
| `build_asos_features.py` | Fetches IEM ASOS observations and aggregates morning features (00:00–10:00 local): temp, dewpoint, RH, pressure, wind, cloud cover. |
| `build_calendar_lag_features.py` | Builds day-of-year harmonics and CLI Tmax lag/rolling features from `cli_target` history. |
| `build_trackB_features.py` | Assembles full Track-B feature tables by joining ASOS, calendar/lags, NWS MOS, GFS, and Open-Meteo NWP sources. |
| `build_trackA_table.py` | Builds Track-A modeling tables from raw feature sources. |
| `fetch_gfs_herbie.py` | Downloads GFS pgrb2.0p25 fields via Herbie; extracts afternoon t2m, dewpoint, and cloud cover with availability-lag guards. |
| `fetch_nws_forecast.py` | Fetches NWS MOS/NBM Tmax forecasts from IEM MOS archive (evening-cycle preference). |
| `fetch_openmeteo_nwp.py` | Fetches ECMWF IFS and GFS seamless daily Tmax from Open-Meteo historical/forecast API. |
| `fetch_cli_target.py` | Fetches and parses NWS CLI daily Tmax products from IEM for lag features and settlement. |
| `predict.py` | Track-J point forecast loading and prediction helpers. |
| `hf_data_store.py` | Upload/download raw Track-J fetch artifacts to HuggingFace dataset `oovanger/MCP_datset`. |

## Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `run_daily_trade.py` | Daily trading pipeline. Run at 10:00 AM CT; fetches market, builds forecasts, selects/sizes trades, logs decisions. |
| `settle_daily.py` | Settle paper trades using NWS CLI Tmax; computes net PnL and appends to `logs/settlements.jsonl`. |
| `build_splits.py` | Build train/test parquet splits from raw Tmax CSV exports. |
| `fetch_recent_market_days.py` | Fetch and append market data for a specific date range via Codex module; merges into per-city CSVs. |
| `fetch_and_validate.py` | Fetch and validate live feature vectors for a date range; runs leakage audit. |
| `build_trackB_all_cities.py` | Build Track-B feature tables for all train cities. |
| `train_trackB_all_cities.py` | Train Track-B ensemble models (Ridge + Huber + LightGBM) for all 9 train cities. |
| `run_baselines_smoke_test.py` | In-sample smoke test for all three core baselines on `threshold_opt.parquet`. |
| `run_oos_evaluation.py` | Out-of-sample evaluation driver for baseline strategies. |
| `run_trackB_grid.py` | Grid search over Track-B signal/sizer/selection configurations. |
| `run_fresh_validation.py` | End-to-end fresh-window backtest on post-OOS market data. |
| `fetch_gfs_all_cities.py` | Batch GFS Herbie download for all deploy cities. |
| `fetch_all_cli_targets.py` | Batch CLI target fetch for all cities. |
| `fetch_all_asos_features.py` | Batch ASOS feature fetch for all cities. |
| `run_nws_forecast_batch.py` | Batch NWS MOS forecast fetch for all cities. |
| `run_openmeteo_nwp_batch.py` | Batch Open-Meteo NWP fetch for all cities. |

## Configuration (`config/`, gitignored)

| File | Description |
|------|-------------|
| `deploy_config.json` | Deployment parameters: signal, sizer, selection rule, edge threshold, cities, bankroll caps. |
| `city_config.json` | Per-city metadata: station codes, lat/lon, timezone, Gaussian sigma for bucket probability conversion. |
| `frozen_params.json` | Frozen baseline threshold parameters (stored under `data/splits/` at runtime). |
| `frozen_k.json` | Stability rule k=1 (stored under `data/splits/` at runtime). |

## Data (gitignored, on HuggingFace)

| Location | Description |
|----------|-------------|
| `data/splits/` | Train/val/test partition parquets (`threshold_opt`, `time_holdout`, etc.) |
| `data/trackb/` | Track-B feature tables and backtest grid results |
| `models/trackb/<city>/` | Per-city trained model artifacts (Ridge, Huber, LightGBM joblib + `feature_cols.json`) |
| `logs/` | Paper/live trade logs (`paper_trades.jsonl`, `settlements.jsonl`) |
| `data/trackj/` | CLI targets, ASOS raw cache, per-city feature parquets |
| `data/raw/` | GFS Herbie CSV caches per station |
| HuggingFace `oovanger/MCP_datset` | Raw market snapshots, NWS/ASOS fetch artifacts |

## Context (`ctx/`)

| File | Description |
|------|-------------|
| `report.md` | Pointer to research log |
| `repo_structure.md` | This file |
| `pipeline_flowchart.md` | End-to-end daily trading pipeline |
| `week3_plan.md` | Current week's operational plan |
| `project_roadmap.md` | Overall goals, priorities, research agenda |

## Tests (`tests/`)

| File | Description |
|------|-------------|
| `test_leakage.py` | Leakage guard tests for feature pipeline. |
| `test_sizing.py` | Kelly sizing and edge guardrail unit tests. |
| `test_track_j.py` | Track-J bucket probability conversion tests. |
