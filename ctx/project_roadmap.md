# Project Roadmap and Research Agenda

Last updated: 2026-06-24 (Go-Live Day 1)

## Mission
Build and deploy a profitable automated trading strategy for Polymarket
same-day Tmax prediction markets. Survive a 60-day live evaluation with
MCP (100 pUSD, 80-trade minimum, 30% max drawdown). Demonstrate
quantitative edge sufficient for a GO hiring decision.

## Current Status: LIVE TRADING -- DAY 1
Track-B deployed as the live strategy. Auto-trader running via cron
in --mode live. First real orders placed Wed Jun 24, 2026.
60-day clock starts today.

Deployed configuration:
- Signal: track_b_flat (Gaussian CDF bucket probabilities, +1.0F Wunderground bias correction)
- Sizer: flat_5 (5 contracts per trade; 3 below $85 bankroll)
- Selection: edge_threshold E*=0.037 on best tradeable bucket per city
- Entry: maker (post_only=True, one tick inside best_ask, 60-min timeout)
- Exit: profit_target_15c (taker at best_bid when gain >= 15c)
- No stop loss (positions fall through to settlement if no exit fires)
- Daily loss cap: $6
- Active cities: Austin (KAUS), Houston (KHOU), LA (KLAX), SF (KSFO)
- VPN: Mexico (required for Polymarket order placement from US)

---

## Priority Framework (revised Jun 24)

S. Pre-challenge sprint: COMPLETE
A. Live trading stability and daily execution (Weeks 1-2)
B. Model improvements: Wunderground retraining, stop loss (Week 2)
C. City expansion and trade count (Weeks 2-4)
D. Strategy refinement: Sequential Bayesian, entry timing (Weeks 3-5)
E. Portfolio optimization, regime detection, scaling (Weeks 5-8)

---

## Pre-Challenge Sprint (Jun 18-23): COMPLETE

### S1. Track-B Extended Backtest
- [x] 83-day Kalshi backtest: Sharpe 7.35, $79.31 PnL, 206 trades, 72.3% win rate
- [x] All 6 MCP GO/NO-GO criteria PASS
- [x] Results documented in research_log.pdf Section 10

### S2. Sequential Bayesian Build
- [x] 5/18 NGBoost feature parquets built (paused, disk space)
- DEFERRED to Week 2-3. Track-B deployed as baseline.

### S3. Polymarket Paper Trading
- [x] 4 paper trading days complete (Jun 20-23)
- [x] Auto-trader tested Mon+Tue with profit_target_15c exits firing both days
- [x] Mon: Houston 92-93F, entry $0.57, exit $0.74, PnL +$0.85
- [x] Tue: LA 70-71F, entry $0.27, exit $0.43, PnL +$0.80

### S4. Resolution Source Verification
- [x] Confirmed: Polymarket uses Wunderground (max of hourly METAR), not NWS CLI
- [x] Systematic +1.0F bias (CLI > Wunderground) confirmed across all 4 active cities
- [x] NYC (KLGA vs KNYC) and Chicago (KORD vs KMDW) excluded due to station mismatch
- [x] Austin and Philadelphia excluded due to insufficient OOS coverage

### S5. GO/NO-GO Decision
- [x] DECISION: GO (Tue Jun 23 evening)
- [x] Cron switched to --mode live
- [x] VPN (Mexico) resolves geo-block; post+cancel verified
- [x] Pushover notifications operational (env var fix: API_KEY -> API_TOKEN)

---

## Week 1 (Jun 24-29): First Live Trades and Stabilization

### 1A. Daily Live Trading [Priority A]
Auto-trader handles entry and exit. Manual oversight 2-3x per day.
- [ ] First live trade placed Wed Jun 24
- [ ] 5+ live trading days completed by Sun Jun 29
- [ ] No API errors or order rejections
- [ ] Fill prices match expected within 2c
- [ ] VPN stable throughout trading hours each day

### 1B. Settlement Tracking [Priority A]
- [ ] Build settle_daily_poly.py for Wunderground-based Polymarket settlement
- [ ] Track cumulative live PnL in logs/poly_settlements.jsonl
- [ ] Daily PnL review and reconciliation
- [ ] Verify Wunderground actuals match Polymarket resolution per city

### 1C. Live Data Collection [Priority A]
Week 1 is primarily about data collection, not returns.
- [ ] Record maker fill rate (what fraction of entries fill within 60 min?)
- [ ] Record slippage (fill price vs intended price)
- [ ] Record exit frequency (profit target fires vs settlement)
- [ ] Gather calibration data: predicted bucket vs actual Wunderground Tmax
- [ ] Target: 10-15 trades by end of Week 1

### 1D. Operational Stability [Priority A]
- [ ] VPN auto-reconnect confirmed stable over multiple days
- [ ] No cron failures or script crashes
- [ ] MacBook clamshell mode sustains cron through trading hours
- [ ] HuggingFace 401 error resolved or suppressed (TRACKJ_SKIP_HF_SYNC=1)

---

## Week 2 (Jun 30 - Jul 5): Wunderground Retraining and City Expansion

### 2A. Wunderground Target Retraining [Priority B, HIGHEST ROI]
Current ensemble is trained on NWS CLI Tmax with a post-hoc +1.0F
correction for Wunderground settlement. More principled: retrain the
ensemble directly on Wunderground Tmax as the target variable.
- [ ] Collect historical Wunderground daily Tmax for all active cities (2021-2025)
- [ ] Retrain Ridge + Huber + LightGBM ensemble on Wunderground target
- [ ] Re-calibrate per-city sigma on Wunderground test-set hit rates
- [ ] Compare retrained model edge vs current bias-corrected model on backtest
- [ ] If retrained model improves edge: deploy and remove post-hoc correction
- [ ] Document results in research log

### 2B. Stop Loss Evaluation [Priority B]
After 5-7 days of live data, evaluate whether adding a stop loss improves
risk-adjusted returns.
- [ ] Tabulate all settlement losses: how many exceed 10c, 15c, 20c?
- [ ] Characterize loss patterns: gradual intraday decay vs flat-then-crash
- [ ] If gradual decay dominates: backtest 10c and 15c stop loss on Kalshi data
- [ ] Decision: add stop loss to auto-trader or keep hold-to-settlement
- [ ] If added: verify stop loss + taker fee interaction (loss = stop + ~6c fee)

### 2C. City Expansion Scoping [Priority B]
With 4 cities averaging ~1 trade/day, pacing is ~60 trades over 60 days.
Need 80. City expansion is the primary lever for trade count.
- [ ] Enumerate ALL Polymarket Tmax cities across all available dates
- [ ] For each candidate: verify ASOS station, Wunderground resolution match
- [ ] Check data availability back to 2021 for training
- [ ] Identify 3-5 Tier 1 expansion cities
- [ ] Begin feature table build for expansion cities

### 2D. Trade Count Pacing [Priority A]
- [ ] Verify cumulative trade count vs 1.33/day minimum
- [ ] If behind pace: lower E* threshold (backtest first) or add cities
- [ ] Consider trading second-best bucket per city if edge exists
- [ ] Monitor that increased frequency does not degrade Sharpe

---

## Week 3 (Jul 6-12): City Deployment and Model Refinement

### 3A. Train and Deploy Expansion Cities [Priority B]
- [ ] Train Track-B ensemble for 3-5 new cities (Wunderground target)
- [ ] Per-city sigma calibration and backtest
- [ ] Exclude any city with negative backtest Sharpe
- [ ] Add to deploy config and go live

### 3B. Realistic Paper Fill Simulation [Priority C]
Improve paper mode fidelity for ongoing validation.
- [ ] Track best_ask each tick; mark fill only when best_ask <= maker_entry_price
- [ ] Report fill rate statistics from paper data
- [ ] Informs whether maker entry strategy needs adjustment

### 3C. Entry Timing Research [Priority C]
Polymarket markets open ~5 days before event. Test earlier entry.
- [ ] Compare D-0 10AM entry vs D-1 evening entry on backtest data
- [ ] Evaluate risk-adjusted edge at each horizon
- [ ] Decision: add D-1 entry or stay with D-0 10AM only

### 3D. Multi-Bucket Trading [Priority C]
Only pursue if trade count is below pace after city expansion.
- [ ] Backtest: trade top-2 and top-3 buckets per city
- [ ] Account for correlation between adjacent buckets
- [ ] Per-date exposure caps must still hold

---

## Week 4 (Jul 13-19): Mid-Challenge Assessment

### 4A. Mid-Challenge Report [Priority A]
After 20 trading days, produce comprehensive assessment:
- [ ] Realized Sharpe vs backtest Sharpe (with bootstrap CIs)
- [ ] Realized vs expected trade count
- [ ] Per-city PnL breakdown
- [ ] Max drawdown and equity curve
- [ ] PnL concentration (top-3 day contribution)
- [ ] Calibration: realized bucket accuracy vs predicted
- [ ] Maker fill rate analysis
- [ ] Adjustments needed for second half

### 4B. Sigma Drift Monitoring [Priority D]
- [ ] Rolling 14-day hit rate tracker per city
- [ ] Alert if rolling hit rate diverges > 10pp from calibrated rate
- [ ] Summer weather may be more predictable (tighten sigma?)
- [ ] Do NOT recalibrate during challenge without documenting

### 4C. Portfolio Correlation Analysis [Priority D]
- [ ] Same-day win/loss correlation matrix across cities
- [ ] Identify correlated loss clusters (same synoptic regime)
- [ ] Adjust per-date exposure caps if needed

---

## Week 5 (Jul 20-26): Sequential Bayesian and Strategy Refinement

### 5A. Sequential Bayesian Build [Priority D]
If Track-B is performing well, begin building the principled upgrade.
- [ ] Complete NGBoost feature tables (t6/t7/t8, all active cities)
- [ ] Train NGBoost distributional regression (CRPS scoring)
- [ ] Cross-lead-time covariance estimation (Ledoit-Wolf on validation set)
- [ ] Kalman filter implementation
- [ ] Calibration validation (PIT, coverage, bucket reliability)
- [ ] Head-to-head backtest vs Track-B on live-period data
- [ ] Deploy only if clearly superior on Sharpe and calibration

### 5B. Exit Strategy Refinement [Priority C]
With 25+ days of live data:
- [ ] How often does profit_target_15c fire vs settlement?
- [ ] Compare realized exit PnL to hold-to-settlement counterfactual
- [ ] If Sequential Bayesian deployed: evaluate posterior-driven exits
- [ ] Adjust profit target or stop loss levels if justified

### 5C. Per-City Edge Analysis [Priority D]
- [ ] Per-city Sharpe with bootstrap CIs
- [ ] Identify net-negative cities for exclusion
- [ ] Cross-validate: does backtest city ranking predict live?

---

## Week 6 (Jul 27 - Aug 2): Optimization

### 6A. Edge Threshold Re-optimization [Priority E]
- [ ] Walk-forward optimization of E* on live data
- [ ] Trade the selectivity-vs-count curve
- [ ] Only change if statistically justified

### 6B. Kelly Sizing Evaluation [Priority E]
If bankroll is stable above $105 for 2+ weeks:
- [ ] Evaluate half-Kelly or quarter-Kelly sizing
- [ ] Backtest Kelly on live data (walk-forward)
- [ ] Implement with 8% cap if Sharpe supports it

### 6C. Market Regime Detection (exploratory) [Priority E]
- [ ] Cluster trading days by NWP feature patterns
- [ ] Test regime-conditional sigma or edge threshold
- [ ] Evaluate whether Sequential Bayesian heteroscedastic sigma covers this
STATUS: Exploratory. Only if time permits and data is sufficient.

### 6D. External Data Sources (exploratory) [Priority E]
- [ ] Evaluate Zeus AI LENS-Cast: 15-min nowcasting, 2km resolution
- [ ] Check pricing and API access (myzeus.ai)
- [ ] Assess value-add over existing ASOS + GFS + ECMWF stack
- [ ] Only integrate if free or clearly ROI-positive for $100 challenge

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
| Jun 21     | Track-B confirmed | Sequential Bayesian deferred; NGBoost stalled at 5/18 |
| Jun 21     | Wunderground bias +1.0F | Systematic CLI > Wunderground confirmed all cities |
| Jun 21     | 4 cities only | NYC/CHI station mismatch, AUS/PHI OOS gap |
| Jun 21     | CLOB order book fix | Was returning dummy 0.01/0.99; commit 9f8bae3 |
| Jun 21     | Auto-trader built | Automated entry + 1-min exit monitoring |
| Jun 22     | Pushover env var fix | PUSHOVER_API_KEY -> PUSHOVER_API_TOKEN |
| Jun 22     | No stop loss (for now) | Backtest validated without one; revisit after live data |
| Jun 23     | VPN Mexico for geo-block | Post+cancel verified through VPN |
| Jun 23     | GO: switch to --mode live | 2 clean paper days, both with profit target exits |
| Jun 24     | Wunderground retraining #1 priority | Highest ROI improvement for Week 2 |
| Jun 24     | City expansion #2 priority | Trade count pacing (4 cities ~1 trade/day) |
| Jun 24     | Sequential Bayesian deferred to Week 5 | Lower risk improvements first |

---

## Key Risk Register (updated Jun 24)

| Risk | Severity | Mitigation | Status |
|------|----------|------------|--------|
| VPN stability during trading | HIGH | Auto-reconnect, monitor daily | Active |
| Trade count pacing (4 cities) | HIGH | City expansion Week 2-3, lower E* | Monitoring |
| PnL concentration (68% in top 3 days) | HIGH | City expansion, portfolio caps | Week 3 |
| No stop loss on live trades | MEDIUM | Monitor loss patterns, add if justified | Week 2 |
| Wunderground bias correction is post-hoc | MEDIUM | Retrain on Wunderground target | Week 2 |
| Resolution source mismatch (NYC/CHI) | MEDIUM | Excluded from trading | Mitigated |
| Convective weather busts | MEDIUM | Heteroscedastic sigma (Sequential Bayesian) | Week 5 |
| Cross-city correlation | MEDIUM | Per-date exposure caps ($6 daily) | Week 4 |
| Maker fill rate uncertainty | MEDIUM | Collecting live data Week 1 | Active |
| Sigma drift in summer | MEDIUM | Rolling hit rate monitor | Week 4 |
| HuggingFace 401 errors | LOW | Does not block execution, cosmetic | Unresolved |
| Bankroll elimination ($70) | LOW | $6 daily cap, conservative sizing | Implemented |
| Private key exposure | LOW | gitignored, env var | Ongoing |

---

## Improvement Priority Ranking (Jun 24)

Ordered by expected ROI for the challenge:

1. Wunderground retraining (Week 2): retrain ensemble on actual resolution target
2. City expansion (Weeks 2-3): more cities = more trades = better pacing
3. Stop loss evaluation (Week 2): data-driven decision after 5-7 live days
4. Realistic paper fill tracking (Week 3): informs maker strategy
5. Sequential Bayesian (Week 5): principled but complex, defer until basics solid
6. External data / Zeus AI (Week 6+): explore pricing, likely deferred

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
| Cities active | 4 | | | | | | | |
| Maker fill rate | | | | | | | | |
| Profit target exits | | | | | | | | |
