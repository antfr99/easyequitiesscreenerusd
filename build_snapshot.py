"""
Build the price snapshot the Streamlit app reads.

Run locally:
    python scripts/build_snapshot.py

Run in CI (see .github/workflows/refresh-snapshot.yml) once a day after the
US close. This is the piece that keeps the app off Yahoo's rate limiter:
the download happens on a schedule with no user waiting on it, and the app
only ever reads a parquet file.

Optional fundamentals (--with-fundamentals) add market cap, P/E and dividend
yield. Those come from Ticker.info, which is one request per symbol, so it
is slow and only ever appropriate inside the scheduled job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import screener_core as core  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def find_universe_csv() -> Path:
    candidates = list(DATA_DIR.glob("*.csv")) + list(ROOT.glob("*.csv"))
    for path in candidates:
        head = pd.read_csv(path, nrows=1)
        if {"Symbol", "Sector", "Industry"}.issubset({c.strip() for c in head.columns}):
            return path
    raise SystemExit("No universe CSV found (needs Symbol, Sector, Industry columns).")


def merge_with_previous(fresh: pd.DataFrame, previous_path: Path) -> pd.DataFrame:
    """
    Carry forward the last good values for any symbol this run failed to
    fetch.

    This is the key to surviving rate limits. Yahoo throttles by IP, and at
    863 symbols a run will sometimes get cut off partway. Without this, one
    throttled run would blank out the app. With it, a partial run simply
    refreshes part of the universe and the rest keeps yesterday's numbers,
    clearly marked stale by the "As Of" column.
    """
    if not previous_path.exists():
        return fresh

    try:
        previous = pd.read_parquet(previous_path)
    except Exception as exc:
        print(f"[warn] could not read previous snapshot: {exc}")
        return fresh

    fresh = fresh.set_index("Symbol")
    previous = previous.set_index("Symbol")

    stale_symbols = fresh.index[fresh["Close"].isna()]
    carry = previous.index.intersection(stale_symbols)
    if len(carry) == 0:
        return fresh.reset_index()

    # Only carry metric columns; Sector / Industry / Name come from the CSV,
    # which is the source of truth and may have been edited since.
    identity = {"Sector", "Industry", "Company Name"}
    cols = [c for c in fresh.columns if c not in identity and c in previous.columns]
    fresh.loc[carry, cols] = previous.loc[carry, cols]

    print(f"Carried forward {len(carry)} symbol(s) from the previous snapshot.")
    return fresh.reset_index()


def add_fundamentals(df: pd.DataFrame, session, pause: float = 0.4) -> pd.DataFrame:
    """
    One Ticker.info call per symbol, deliberately throttled. Expect roughly
    10 minutes for 863 symbols. Failures are tolerated and left as NaN.
    """
    import yfinance as yf

    rows = []
    total = len(df)
    for i, sym in enumerate(df["Symbol"], start=1):
        record = {"Symbol": sym}
        try:
            info = yf.Ticker(sym, session=session).info or {}
            record["Market Cap"] = info.get("marketCap")
            record["Trailing P/E"] = info.get("trailingPE")
            record["Forward P/E"] = info.get("forwardPE")
            record["Dividend Yield %"] = (
                info.get("dividendYield") * 100
                if isinstance(info.get("dividendYield"), (int, float))
                else None
            )
        except Exception as exc:
            print(f"[warn] info failed for {sym}: {exc}")
        rows.append(record)

        if i % 50 == 0:
            print(f"  fundamentals {i}/{total}")
        time.sleep(pause)

    return df.merge(pd.DataFrame(rows), on="Symbol", how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=core.DEFAULT_HISTORY_DAYS)
    parser.add_argument("--batch-size", type=int, default=core.DEFAULT_BATCH_SIZE)
    parser.add_argument("--pause", type=float, default=core.DEFAULT_PAUSE)
    parser.add_argument("--with-fundamentals", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only fetch N symbols (testing).")
    parser.add_argument(
        "--shard",
        default="",
        help="Fetch only part of the universe, as i/n (e.g. 2/4). Lets you "
             "spread 863 symbols across several runs to stay under Yahoo's "
             "hourly ceiling. Untouched symbols keep their previous values.",
    )
    args = parser.parse_args()

    csv_path = find_universe_csv()
    print(f"Universe: {csv_path.name}")

    universe = core.load_universe(csv_path)
    full_universe = universe.copy()

    if args.limit:
        universe = universe.head(args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        universe = universe.iloc[(i - 1) :: n]
        print(f"Shard {i}/{n}: {len(universe)} symbols")

    symbols = universe["Symbol"].tolist()
    print(f"Symbols to fetch: {len(symbols)} (universe is {len(full_universe)})")

    session = core.make_session()
    if session is None:
        print("[warn] curl_cffi unavailable — expect a higher chance of throttling.")

    started = time.time()
    frames = core.download_prices(
        symbols,
        days=args.days,
        batch_size=args.batch_size,
        pause=args.pause,
        session=session,
        progress_cb=lambda done, total, label: print(f"  {label}"),
    )
    print(f"Pass 1: {len(frames)}/{len(symbols)} in {time.time() - started:.0f}s")

    # Second pass for whatever came back empty, slower and single-threaded-ish.
    missed = [s for s in symbols if s not in frames]
    if missed:
        print(f"Retrying {len(missed)} symbol(s) after a cool-off…")
        time.sleep(30)
        retry = core.download_prices(
            missed,
            days=args.days,
            batch_size=max(10, args.batch_size // 3),
            pause=max(3.0, args.pause * 3),
            session=session,
            progress_cb=lambda done, total, label: print(f"  retry {label}"),
        )
        frames.update(retry)
        print(f"Pass 2 recovered {len(retry)}/{len(missed)}")

    # Always build against the FULL universe so the snapshot keeps every row,
    # then fill the gaps from the previous snapshot.
    df = core.build_metrics_frame(full_universe, frames)
    df = merge_with_previous(df, DATA_DIR / "snapshot.parquet")

    if "As Of" in df.columns and df["As Of"].notna().any():
        newest = pd.to_datetime(df["As Of"]).max()
        df["Stale"] = pd.to_datetime(df["As Of"]) < newest

    if args.with_fundamentals:
        print("Fetching fundamentals (slow)…")
        df = add_fundamentals(df, session)

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "snapshot.parquet"
    df.to_parquet(out, index=False)

    covered = int(df["Close"].notna().sum())
    stale = int(df["Stale"].sum()) if "Stale" in df.columns else 0
    meta = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "universe_size": len(full_universe),
        "symbols_fetched_this_run": len(symbols),
        "symbols_with_data": covered,
        "symbols_carried_forward": stale,
        "shard": args.shard or "all",
        "missing": sorted(df.loc[df["Close"].isna(), "Symbol"].tolist()),
        "history_days": args.days,
        "fundamentals": bool(args.with_fundamentals),
    }
    (DATA_DIR / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out} — {covered}/{len(full_universe)} symbols with data "
          f"({stale} carried forward from the previous snapshot)")
    if meta["missing"]:
        print("Missing:", ", ".join(meta["missing"][:25]),
              "…" if len(meta["missing"]) > 25 else "")


if __name__ == "__main__":
    main()
