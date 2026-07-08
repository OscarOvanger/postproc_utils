# Polymarket v5 Operations Runbook

## Cron not firing on macOS

Cron on macOS requires **Full Disk Access** for `/usr/sbin/cron`.

1. Open **System Settings → Privacy & Security → Full Disk Access**
2. Add `/usr/sbin/cron` (use Go to Folder: `/usr/sbin/cron`)
3. Install the probe line:
   ```bash
   scripts/cron_probe install
   # copy the printed line into: crontab -e
   ```
4. Wait 2+ minutes, then verify:
   ```bash
   scripts/cron_probe check
   ```
5. Remove the probe line (`scripts/cron_probe remove`), then confirm autotrader:
   ```bash
   tail -f logs/cron_stderr.log
   ```
   At the next `:05` tick between 10:00–22:00 CT you should see:
   `=== cron_autotrader.sh ... ===`

## Trade watchdog under cron

To confirm Pushover alerts fire from cron (not manually):

1. Ensure probe is removed and live crontab is installed (`scripts/crontab_v5.txt`)
2. Temporarily stop autotrader from running (e.g. rename venv) or wait past 10:25 without a state file
3. Watchdog at `25 10 * * *` should append to `logs/cron_watchdog.log` and send Pushover

## Lock semantics (macOS)

`scripts/cron_autotrader.sh` uses `mkdir logs/cron_autotrader.lockdir` (atomic).
Stale locks older than 60 minutes are cleared with a stderr alert.

## Morning preflight

```bash
.venv/bin/python scripts/poly_portfolio_status.py --no-forecasts
```

Shows bankroll reconciliation, rolling-bias coverage, strategy/model config, and market availability.
