# Project Roadmap and Research Agenda

Last updated: 2026-06-18 (Pre-Challenge Sprint, Day 1)

## Mission
Build and deploy a profitable automated trading strategy for Polymarket
same-day Tmax prediction markets. Survive a 60-day live evaluation with
MCP (100 pUSD, 80-trade minimum, 30% max drawdown). Demonstrate
quantitative edge sufficient for a GO hiring decision.

## Current Status: PRE-CHALLENGE SPRINT
Two candidate strategies competing in a head-to-head 60-day Kalshi backtest:
- Strategy A (Track-B): proven baseline, OOS Sharpe 10.5, pipeline ready
- Strategy B (Sequential Bayesian): principled upgrade, unbuilt, sprint target
First live trade: Wednesday Jun 24, 2026. 60-day clock starts then.

---

## Priority Framework (revised Jun 18)

S. Pre-challenge sprint: dual-strategy bake-off (Jun 18-23)
A. Live trading stability and daily execution (Weeks 1-2)
B. City expansion and trade count (Weeks 2-4)
C. Strategy refinement: entry timing, multi-horizon (Weeks 3-5)
D. Model improvements: regime detection, sigma monitoring (Weeks 4-6)
E. Portfolio optimization and scaling (Weeks 5-8)

---

## Pre-Challenge Sprint (Jun 18-23)

### S1. Track-B Extended Backtest [Priority S, Day 1]
Run Track-B + profit_target_15c over the full available Kalshi market
window (~77 days, Mar 30 to Jun 14) under MCP constraints.
This is the benchmark. Sequential Bayesian must beat it to deploy.
- [x] 27-day MCP simulation complete (Sharpe 6.20, $25.13 PnL)
- [ ] Extended to full available window with intraday exit
- [ ] Results documented with equity curve and per-city breakdown
STATUS: Queued for Thu Jun 18.

### S2. Sequential Bayesian Build [Priority S, Days 1-4]
Build NGBoost distributional regression + Kalman filter pipeline.
- [ ] Multi-lead-time feature tables (t6/t7/t8, 6 cities)
- [ ] Climatological prior table (10+ year CLI history)
- [ ] NGBoost training (18 models, CRPS scoring)
- [ ] Cross-lead-time covariance estimation (Ledoit-Wolf)
- [ ] Kalman filter implementation + batch posterior verification
- [ ] Calibration validation (PIT, coverage, bucket reliability)
- [ ] 60-day Kalshi backtest under identical MCP constraints
STATUS: Not started. Sprint target: backtest-ready by Sun Jun 21.

### S3. Polymarket Paper Trading [Priority S, Daily]
Run Track-B paper trades on Polymarket every day (Jun 18-23).
Validates pipeline, tests API reliability, builds fill rate data.
- [ ] >= 4 of 6 paper days complete without error
- [ ] Feature coverage >= 5/6 cities each day
- [ ] No wildly incorrect forecasts
STATUS: Pipeline ready. Jun 17 test run succeeded.

### S4. Resolution Source Verification [Priority S, Day 5]
Verify NWS CLI settlement source for all 6 Polymarket Tier 1 cities.
Show-stopper if any city uses Wunderground or a different source.
- [ ] All 6 cities verified
- [ ] Document in docs/polymarket_resolution_sources.md
STATUS: Not started. Critical pre-live task.

### S5. GO/NO-GO Decision [Priority S, Day 6]
Select deployment strategy based on bake-off results.
Deploy the winner. If tie, Track-B wins (proven > unproven).
- [ ] Head-to-head comparison documented
- [ ] deploy_config_poly.json finalized
- [ ] Go-live gate checklist passed
STATUS: Decision on Tue Jun 23 evening.

---

## Week 1 (Jun 24-29): First Trades and Stabilization

### 1A. Daily Live Trading [Priority A]
Execute live Polymarket trades daily using the chosen strategy.
Start conservative: single highest-edge city, 5 contracts max.
Manually verify each order on Polymarket UI after API placement.
- [ ] First live trade placed Wed Jun 24
- [ ] 5+ live trading days completed by Sun Jun 29
- [ ] No API errors or order rejection issues
- [ ] Fill prices match expected prices within 2c
- [ ] Post_only confirmed (maker, zero fees)

### 1B. Settlement Tracking [Priority A]
Build or adapt settlement pipeline for Polymarket.
- [ ] Polymarket settlement script (settle_daily_poly.py)
- [ ] Verify CLI Tmax matches Polymarket resolution for each city
- [ ] Track cumulative P&L in logs/poly_settlements.jsonl
- [ ] Daily P&L review

### 1C. Market Monitor for Polymarket [Priority A]
Adapt market_monitor.py for Polymarket positions.
- [ ] Fetch current Polymarket prices for open positions
- [ ] Evaluate exit conditions (profit target / posterior update / stop loss)
- [ ] Pushover alerts for exit signals
- [ ] Cron schedule (*/5 10-23)

### 1D. Phased Scaling Plan [Priority A]
Per Pro model review, Week 1 is tiny sizing:
- Max 5 contracts per trade (~$1.75 risk at 35c entry)
- Max 2-3 trades per day
- Focus on data collection, not returns
- Target: 10-15 trades by end of Week 1
- Gather live fill/slippage/calibration data

---

## Week 2 (Jun 30 - Jul 5): Scale to Full Operation

### 2A. Expand to All Verified Cities [Priority A]
After 1 week of live data, verify that all 6 cities produce
reasonable forecasts and fills. Drop any city with systematic
problems (resolution mismatch, persistent API issues).
- [ ] Per-city live accuracy tracker (rolling 7-day)
- [ ] Remove or flag underperforming cities
- [ ] Increase to full city coverage

### 2B. Increase Trade Frequency [Priority A]
Week 1 target was 10-15 trades. Week 2 target: 20-25 trades.
Need 80 trades in 60 days = 1.33/day minimum.
- [ ] Verify trade count pacing
- [ ] If behind pace, consider:
  * Lowering edge threshold (if backtest supports)
  * Adding small maker fill orders for count
  * Trading multiple buckets per city if edges exist
- [ ] Monitor that increased frequency does not degrade Sharpe

### 2C. City Expansion Scoping [Priority B]
Enumerate ALL Polymarket Tmax cities (not just today's markets).
For each new city:
  * Verify ASOS station, CLI resolution source
  * Check data availability back to 2021
  * Assess: worth training a new model?
- [ ] Full city inventory documented
- [ ] Tier 1 expansion candidates identified (5-10 new cities)
- [ ] Begin training Track-B (or NGBoost) for new cities

---

## Week 3 (Jul 6-12): City Expansion and Entry Timing

### 3A. Train Models for New Cities [Priority B]
Train the chosen strategy's model for Tier 1 expansion cities.
Same train/val/test split. Calibrate sigma (Track-B) or validate
CRPS (Sequential Bayesian). Backtest on available Kalshi data if
the city overlaps. Include in deploy_config_poly.json.
- [ ] 5-10 new city models trained
- [ ] Per-city backtest results documented
- [ ] Exclude any city with negative backtest Sharpe
- [ ] Live deployment of new cities

### 3B. Entry Timing Research [Priority C]
With multi-day Polymarket horizons (markets open D-5), test whether
entering earlier offers larger edges despite wider uncertainty.
- [ ] Compare D-0 10AM entry vs D-1 evening entry on backtest data
- [ ] If using Sequential Bayesian: natural fit (t6 is D-1 evening)
- [ ] If using Track-B: build a D-1 variant with available features
- [ ] Evaluate risk-adjusted edge at each horizon
- [ ] Decision: add D-1 entry or stay with D-0 10AM only

### 3C. Multi-Bucket Trading [Priority C]
Current strategy trades only the single highest-edge bucket per city.
Explore whether trading 2-3 buckets per city increases Sharpe.
- [ ] Backtest: trade top-2 and top-3 buckets per city
- [ ] Account for correlation between adjacent buckets
- [ ] Per-date exposure caps must still hold
STATUS: Deferred. Only pursue if trade count is below pace.

---

## Week 4 (Jul 13-19): Mid-Challenge Assessment

### 4A. Mid-Challenge Report [Priority A]
After 20 trading days, produce a comprehensive assessment:
- [ ] Realized Sharpe vs backtest Sharpe
- [ ] Realized vs expected trade count
- [ ] Per-city P&L breakdown
- [ ] Max drawdown and equity curve
- [ ] PnL concentration (top-3 day contribution)
- [ ] Calibration: realized bucket accuracy vs predicted
- [ ] Comparison to baseline strategies
- [ ] Adjustments needed for second half

### 4B. Sigma Drift Monitoring [Priority D]
- [ ] Rolling 14-day hit rate tracker per city
- [ ] Alert if rolling hit rate diverges > 10pp from calibrated rate
- [ ] Summer weather may be more predictable (tighten sigma?)
- [ ] Do NOT recalibrate during challenge without documenting

### 4C. Portfolio Correlation Analysis [Priority D]
Review live trading data for cross-city correlation:
- [ ] Same-day win/loss correlation matrix
- [ ] Identify correlated loss clusters (same weather regime)
- [ ] Adjust per-date exposure caps if needed

---

## Week 5 (Jul 20-26): Strategy Refinement

### 5A. Per-City Edge Analysis [Priority D]
With 25+ trading days of live data:
- [ ] Per-city Sharpe with bootstrap CIs
- [ ] Identify net-negative cities for exclusion
- [ ] Adjust per-city edge thresholds if justified
- [ ] Cross-validate: does backtest city ranking predict live?

### 5B. Exit Strategy Refinement [Priority C]
Review live exit performance:
- [ ] How often does profit_target_15c fire?
- [ ] How often does posterior-driven exit fire (if deployed)?
- [ ] Compare realized exit P&L to hold-to-settlement counterfactual
- [ ] Adjust profit target or stop loss levels if justified

### 5C. Kelly Sizing Evaluation [Priority E]
If bankroll is stable above $105 for 2+ weeks:
- [ ] Evaluate half-Kelly or quarter-Kelly sizing
- [ ] Backtest Kelly on live data (walk-forward)
- [ ] Implement with 8% cap if Sharpe supports it
STATUS: Only if bankroll is healthy. Flat sizing is safe.

---

## Week 6 (Jul 27 - Aug 2): Optimization

### 6A. Edge Threshold Re-optimization [Priority E]
With 30+ days of live data, re-evaluate E*:
- [ ] Walk-forward optimization of E* on live data
- [ ] Trade the selectivity-vs-count curve
- [ ] Only change if statistically justified

### 6B. Market Regime Detection (exploratory) [Priority D]
- [ ] Cluster trading days by NWP feature patterns
- [ ] Test regime-conditional sigma or edge threshold
- [ ] Requires enough data to estimate seasonal effects
STATUS: Exploratory. Only if time permits and data is sufficient.

---

## Week 7 (Aug 3-9): Final Stretch

### 7A. Sharpe Pacing Review [Priority A]
- [ ] Current Sharpe trajectory vs target
- [ ] Trade count pacing (must hit 80 by Day 60)
- [ ] If behind on trades: lower threshold or add cities
- [ ] If behind on Sharpe: reduce risk, focus on consistency

### 7B. Final Portfolio Adjustments [Priority E]
- [ ] Remove any city with negative live Sharpe
- [ ] Tighten sizing if drawdown is concerning
- [ ] Avoid open unresolved tail risk near end of challenge

---

## Week 8 (Aug 10-18): Close-Out

### 8A. Final Trading Days [Priority A]
- [ ] Continue daily execution
- [ ] Avoid large new positions in last 3 days
- [ ] Let existing positions settle normally
- [ ] Ensure trade count >= 80

### 8B. Final Report [Priority A]
- [ ] Comprehensive performance report for MCP
- [ ] Realized vs backtest comparison
- [ ] Per-city, per-week breakdown
- [ ] Strategy description (suitable for MCP evaluation)
- [ ] Lessons learned and recommendations

### 8C. Career Follow-Up [Priority A]
- [ ] Package results for G-Research conversation with Leon Dimas
- [ ] Update CV/resume with challenge results
- [ ] Norwegian contacts follow-up (Ine Oma, Trym Steinset Torvund)
- [ ] Applications to target firms if results warrant

---

## Decision Log

| Date       | Decision | Rationale |
|------------|----------|-----------|
| Jun 7      | Track-B over Track-A | NWP features are the differentiator |
| Jun 10     | Flat sizing over Kelly | Conservative for early bankroll |
| Jun 13     | GO decision | 6/7 criteria passed, Sharpe 10.95 |
| Jun 14     | Intraday exit confirmed | Kalshi allows contract sale before settlement |
| Jun 14     | Repo made public | Renamed postproc_utils, sensitive files scrubbed |
| Jun 15     | Pipeline restructured | Features pre-fetched before market wait |
| Jun 15     | Entry timing deferred | Backtest evidence is at 10AM; don't change before live |
| Jun 15     | City expansion elevated | Highest-leverage improvement after exit strategy |
| Jun 16     | profit_target_15c selected | OOS Sharpe 11.85 vs 10.47 hold-to-settlement |
| Jun 16     | Market monitor deployed | Cron */5 10-23, commit 81570ee |
| Jun 16     | ASOS fallback deployed | Fixes LA/PHX/SF coverage, commit d5000a4 |
| Jun 17     | Polymarket pivot | MCP uses Polymarket, not Kalshi |
| Jun 17     | Dual-track trading | Kalshi paper + Polymarket live in parallel |
| Jun 18     | Dual-strategy bake-off | Track-B vs Sequential Bayesian on 60-day Kalshi data |
| Jun 18     | First live trade Wed Jun 24 | 6 days paper + build time |
| Jun 18     | Conservative sizing | 2-5 pUSD max loss per trade (Pro model review) |
| Jun 18     | Climatological prior | Avoids NWP double-counting in market prices |
| Jun 18     | t6/t7/t8 only | Pragmatic: 18 models vs 72. Extend later if justified |
| Jun 18     | Resolution verification | Non-negotiable before live (Pro model review) |

---

## Key Risk Register (updated Jun 18)

| Risk | Severity | Mitigation | Status |
|------|----------|------------|--------|
| PnL concentration (68% in top 3 days) | HIGH | City expansion, portfolio caps | Sprint + Week 3 |
| Resolution source mismatch | HIGH | Per-city verification before live | Sprint Day 5 |
| Convective weather busts | HIGH | Heteroscedastic sigma, exit rules | Sequential Bayesian |
| Sequential Bayesian build failure | HIGH | Track-B is always the fallback | Sprint |
| Polymarket API reliability | MEDIUM | Paper trading + error handling | Sprint |
| Cross-city correlation | MEDIUM | Per-date exposure caps | Week 4 |
| Maker fill rate uncertainty | MEDIUM | Monitor during paper trading | Sprint |
| Per-city negative edge | MEDIUM | Per-city analysis Week 5 | Week 5 |
| Sigma drift in summer | MEDIUM | Rolling hit rate monitor | Week 4 |
| 80-trade pacing risk | LOW | 6 cities x 60 days = ~120 potential | Comfortable |
| Bankroll elimination ($70) | LOW | $6 daily cap, conservative sizing | Implemented |
| Sprint time pressure | LOW | Track-B fallback if NGBoost delayed | Sprint |
| Private key exposure | LOW | gitignored, env var recommended | Ongoing |

---

## Metrics Dashboard (to be populated weekly from Week 1)

| Metric | Week 1 | Week 2 | Week 3 | Week 4 | Week 5 | Week 6 | Week 7 | Week 8 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Cumulative PnL | | | | | | | | |
| Bankroll | | | | | | | | |
| Sharpe (running) | | | | | | | | |
| Sortino (running) | | | | | | | | |
| Max drawdown | | | | | | | | |
| Total trades | | | | | | | | |
| Win rate | | | | | | | | |
| Trade count pace | | | | | | | | |
| Cities active | | | | | | | | |
