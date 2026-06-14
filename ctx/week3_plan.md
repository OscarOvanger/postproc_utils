# Week 3 Plan (Jun 14-20, 2026)

## Decision: GO
Leading config: track_b_flat + flat_5 + edge_threshold (E*=0.037)
Combined 25-day Sharpe: 10.95 [7.89, 14.01], 68 trades, max DD $1.58

## Track 1: Paper Trading → Live Trading

### Paper Trading Phase (Jun 14-18, target 5 clean days)
- Run run_daily_trade.py --mode paper at 10:00 AM CT daily
- Run settle_daily.py the following morning
- Success criteria for going live:
  1. Pipeline completes without errors for 3 consecutive days
  2. Feature coverage >= 70% of cities each day
  3. No wildly incorrect forecasts (manual sanity check)
  4. Paper PnL is not catastrophically negative (> -$10 cumulative)
- If all criteria met by Wed Jun 18: first live trade Thu Jun 19
- 60-day clock starts on first live trade

### Live Trading Phase (Jun 19+ if paper is clean)
- Daily routine:
  08:00 AM CT  Verify GFS/ECMWF data availability
  09:30 AM CT  Run feature pipeline, check for missing data
  10:00 AM CT  Run run_daily_trade.py --mode live
  10:05 AM CT  Verify orders placed on Kalshi
  11:00 PM CT  Run settle_daily.py (or next morning)
- Start with $100 USDC on Kalshi
- Oscar may also place small discretionary trades ($5 max) during
  paper phase to test the Kalshi API flow

### Risk Monitoring
- Daily: check cumulative PnL, max drawdown, trade count pace
- If bankroll < $85: reduce to 3 contracts (automatic)
- If bankroll < $75: pause and review (manual gate)
- If 3 consecutive losing days: review forecast accuracy, do not
  change parameters but document the drawdown
- Weekly: compare realized Sharpe to backtest expectation

## Track 2: Research (parallel, lower priority)

### Deferred improvements (do NOT implement during paper trading)
- NBM uncertainty quantiles replacing Gaussian sigma assumption
- Bayesian posteriors for Kelly sizing
- GraphCast/Pangu-Weather downscaling as additional signal
- HRRR/RAP features for non-Austin cities
- Walk-forward retraining as more data accumulates

### Data maintenance
- Rebuild splits daily (python scripts/build_splits.py)
- Upload new market data to HuggingFace weekly
- Monitor NWS CLI availability (Philadelphia had gaps Jun 11-12)

## Track 3: Career / Networking
- Follow up with MCP on Quant Analyst application
- G-Research: coordinate with Leon Dimas before applying
- Norwegian contacts: send drafted outreach messages
