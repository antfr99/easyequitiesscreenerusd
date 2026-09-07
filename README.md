# easyequitiesscreenerusd

Streamlit screener over the EasyEquities USD universe (~863 tickers), with
ticker, sector and industry views.

## Layout

```
streamlit_app.py                     the app
screener_core.py                     shared data layer (no Streamlit imports)
scripts/build_snapshot.py            builds data/snapshot.parquet
data/USD Easy Equities with Sectors.csv   universe: Symbol, Sector, Industry, Company Name
data/snapshot.parquet                generated — the app reads this
data/snapshot_meta.json              generated — build time and coverage
.github/workflows/refresh-snapshot.yml    daily rebuild
requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
python scripts/build_snapshot.py --limit 25   # quick smoke test
streamlit run streamlit_app.py
```

Without a snapshot the app falls back to a live pull, cached for an hour.
That's fine for a handful of tickers and slow for the full universe.

## Why a snapshot instead of live calls

Yahoo rate limits by IP. `yf.download` issues one request per symbol even
when you pass a list, so the full universe is ~863 requests, and users have
reported 429s in that same range. Three things keep this workable:

1. **Batching with a pause** — 60 symbols per batch, one second between
   batches, `curl_cffi` browser impersonation, and `retries = 2`.
2. **A retry pass** — anything that comes back empty is refetched after a
   30-second cool-off with smaller batches and longer pauses.
3. **Carry-forward** — symbols that still fail keep their previous values
   and are flagged `Stale`, so a throttled run degrades instead of breaking.

If the nightly run gets throttled consistently, split it:

```bash
python scripts/build_snapshot.py --shard 1/4    # then 2/4, 3/4, 4/4
```

Each shard refreshes a quarter of the universe and leaves the rest intact.

Note that GitHub Actions runners use shared datacenter IPs, which Yahoo
throttles more aggressively than a home connection. If CI keeps coming back
half-empty, run `build_snapshot.py` on a local schedule and push the parquet
instead.

## Sector and industry

These come from the CSV, not from Yahoo. `Ticker.info` does expose `sector`
and `industry`, but it's one request per symbol on the heavier
`quoteSummary` endpoint — the fastest way to get rate limited. The CSV is
the source of truth and always wins over anything in the snapshot.

## Metrics

Everything below is derived from a single daily price history download:
close, closes and % change over 1d / 1w / 4w / 13w / 26w / 52w, SMA 20 / 50 /
200 and distance from each, 52-week high and low, annualised 1-month realised
volatility, and 20-day average volume and dollar volume.

Lookbacks resolve to the last trading day on or before the target date, so
the "4 weeks ago" close is a real close and not a holiday gap.

Market cap, P/E and dividend yield need `Ticker.info`, so they're opt-in via
`--with-fundamentals` and only ever run inside the scheduled job.

## Branch workflow

`main` stays deployable — it's what Streamlit Cloud serves.

```bash
git checkout main
git pull
git checkout -b feature/industry-heatmap

# ... work, committing as you go ...
git add -A
git commit -m "Add industry heatmap tab"
git push -u origin feature/industry-heatmap
```

Then open a pull request on GitHub, review your own diff, and merge with
**Squash and merge** so `main` keeps one commit per feature.

```bash
git checkout main
git pull
git branch -d feature/industry-heatmap
```

To see a branch running before merging, deploy a second Streamlit app from
the same repo pointed at the branch. You get a staging URL, and `main` is
untouched.

Worth setting once, under Settings → Branches → Add rule for `main`: require
a pull request before merging. It stops a tired `git push` straight to main.

Note the deploy form defaults the branch to `master`; this repo uses `main`,
so change it. Main file path is `streamlit_app.py`.
