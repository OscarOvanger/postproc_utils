# Week 1 Daily Plan (Jun 24-29)
# Created Wed Jun 24 (Go-Live Day 1)

## Wednesday Jun 24 -- GO-LIVE (today)

### Morning
- [x] VPN verified (Mexico, curl ifconfig.me)
- [x] Cron set to --mode live
- [ ] First live trade notification received via Pushover
- [ ] Check logs after 10:15 AM:
    tail -30 logs/auto_trader.log
    cat logs/auto_trader_state_2026-06-24.json
- [ ] Verify order appears on Polymarket (check via API or web UI through VPN)

### Intraday
- Check logs every 1-2 hours
- Watch for profit_target_15c exit notification
- If anything looks wrong: comment out cron line (crontab -e)
- Do NOT manually intervene unless there is an error

### Evening
- [ ] Auto-trader end-of-day summary at 10 PM CT
- [ ] Review state file: entries, exits, monitoring log
- [ ] Check Wunderground for today's actuals (if available)
- [ ] Record: how many cities traded? Fill rate? Any exits?
- [ ] Commit state file and logs

### Also today
- [ ] Settle Jun 22 Kalshi paper trades:
    .venv/bin/python scripts/settle_daily.py --date 2026-06-22
- [ ] Settle Jun 23 Kalshi paper trades:
    .venv/bin/python scripts/settle_daily.py --date 2026-06-23
- [ ] Run Kalshi paper trade (parallel track):
    .venv/bin/python scripts/run_daily_trade.py --date 2026-06-24 --mode paper --bankroll 100.00

---

## Thursday Jun 25 -- Live Day 2

### Morning (before 10:00 AM CT)
- Verify VPN is connected:
    curl -s ifconfig.me
  If US IP: reconnect VPN before 10:00
- Auto-trader fires at 10:05 AM, no manual action needed
- Check Pushover for entry notification around 10:10

### After 10:15 AM
- Review logs:
    tail -30 logs/auto_trader.log
    cat logs/auto_trader_state_2026-06-25.json
- Confirm: how many entries? Which cities? Entry prices?
- Compare to yesterday's pattern

### PM Block
- Settle yesterday's Polymarket trades (manual Wunderground check):
    Check wunderground.com for Jun 24 Tmax at KAUS, KHOU, KLAX, KSFO
    Compare to model predictions
    Record in a simple log (city, predicted bucket, actual Tmax, win/loss)
- Settle Jun 24 Kalshi paper trades:
    .venv/bin/python scripts/settle_daily.py --date 2026-06-24

### Evening
- Review auto-trader end-of-day summary
- Calculate running stats:
    Live trades so far: ?
    Live PnL so far: ?
    Maker fill rate: filled / attempted
    Profit target exits: count
    Settlement outcomes: count
- Run Kalshi paper trade for tomorrow if not automated

### Development (if time permits)
- Start scoping Polymarket settlement script (settle_daily_poly.py)
  Needs to: fetch Wunderground Tmax, match to position bucket, compute PnL
- Or: begin city expansion inventory (which Polymarket cities exist?)

---

## Friday Jun 26 -- Live Day 3

### Morning
- VPN check: curl -s ifconfig.me
- Auto-trader runs at 10:05
- Check Pushover, then logs after 10:15

### PM Block
- Settle Jun 25 Polymarket trades (Wunderground manual check)
- Settle Jun 25 Kalshi paper trades
- Review 3-day running stats

### Development (pick one focus area)

Option A: Polymarket settlement automation
- Build settle_daily_poly.py that fetches Wunderground Tmax per city
- Input: auto_trader_state file with settlement_pending positions
- Output: append to logs/poly_settlements.jsonl
- Priority: needed before trade volume grows

Option B: City expansion inventory
- Query Gamma API for all temperature markets across all dates
- List every city Polymarket has ever offered
- Cross-reference with our ASOS station list
- Flag which cities we already have models for
- Flag which need new training data

### Evening
- End-of-day summary review
- 3-day performance snapshot
- Plan which development track to push on Saturday

---

## Saturday Jun 27 -- Live Day 4

### Morning
- VPN check, auto-trader runs at 10:05, review logs

### PM Block
- Settle Jun 26 (both platforms)
- Continue whichever development track started Friday
  (settlement script OR city inventory)

### Development
If settlement script done: start city inventory
If city inventory done: begin Wunderground historical data collection
  for retraining (2021-2025, 4 active cities)

### Evening
- 4-day running stats
- Check trade count pacing: on track for 10-15 by Sunday?
- If 0-1 trades/day: consider lowering E* from 0.037 to 0.030
  (backtest this on Kalshi data before changing live config)

---

## Sunday Jun 28 -- Live Day 5

### Morning
- VPN check, auto-trader runs at 10:05, review logs
- Note: Polymarket Tmax markets may have reduced liquidity on weekends
  Watch for wider spreads or empty order books

### PM Block
- Settle Jun 27 (both platforms)
- Complete any outstanding development from Sat

### Week 1 Review
Compile Week 1 performance summary:
- Total live trades placed
- Total live trades filled (maker fill rate)
- Profit target exits: count and total PnL
- Settlement outcomes: count, wins, losses, total PnL
- Cumulative PnL and bankroll
- Per-city breakdown
- Any API errors, crashes, or VPN issues
- Trade count pacing: projected 60-day total at current rate

### Planning
- Write Week 2 daily plan
- Decide: is trade count pacing acceptable? If not, what to change
- Decide: start Wunderground retraining or city expansion first?
- Set up any needed Cursor prompts for Monday development

### Evening
- Commit and push all state files, logs, and any new scripts
- Message Hector with a brief Week 1 update (if results are worth sharing)

---

## Monday Jun 29 -- Live Day 6 (end of Week 1)

### Morning
- VPN check, auto-trader runs at 10:05

### PM Block
- Settle Jun 28 (both platforms)
- Finalize Week 1 review document
- Update project_roadmap.md with Week 1 metrics

### Development: Begin Week 2 Priority #1
Start whichever is more urgent based on Week 1 data:

If trade count is below pace (< 8 trades in Week 1):
  -> City expansion is #1 priority
  -> Start training models for 2-3 new cities

If trade count is on pace but losses are high:
  -> Wunderground retraining is #1 priority
  -> Begin historical Wunderground data collection

If both are fine:
  -> Wunderground retraining (still highest ROI)
  -> City expansion scoping in parallel

---

## Daily Routine Checklist (every trading day)

Before 10:00 AM CT:
  [ ] VPN connected (curl -s ifconfig.me -> non-US IP)
  [ ] MacBook plugged in, lid closed (clamshell mode)

After 10:15 AM CT:
  [ ] Check Pushover for entry notification
  [ ] tail -20 logs/auto_trader.log
  [ ] cat logs/auto_trader_state_$(date +%Y-%m-%d).json | python3 -m json.tool

Intraday:
  [ ] Glance at Pushover 1-2x for exit signals
  [ ] No manual intervention unless errors

Evening:
  [ ] Review end-of-day summary (10 PM CT Pushover)
  [ ] Settle previous day's trades (Wunderground check)
  [ ] Update running PnL tracker
  [ ] Note any issues for tomorrow

---

## Week 1 Targets

| Target | Threshold | Action if missed |
|--------|-----------|------------------|
| Total trades >= 8 | Critical (80/60 = 1.33/day) | Lower E* or add cities Week 2 |
| No API errors | Critical | Debug immediately |
| VPN stable all 6 days | Critical | Find alternative VPN or server |
| At least 1 profit target exit | Important | Verify exit logic on live data |
| Maker fill rate > 50% | Important | Adjust entry pricing (tighter spread) |
| No bankroll below $94 | Comfortable | Review if breached |
| Settlement script built | Nice to have | Finish early Week 2 |
