"""
Shared data layer for the EasyEquities USD screener.

Used by both streamlit_app.py (live mode) and scripts/build_snapshot.py
(scheduled mode). Everything in here is plain pandas/yfinance with no
Streamlit imports, so it can run inside a GitHub Action.

Design note: every metric below is derived from ONE bulk price-history
download. Nothing here calls Ticker.info, because that is one HTTP request
per ticker and is what triggers Yahoo rate limiting at 863 symbols.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = ["Symbol", "Sector", "Industry", "Company Name"]

TRADING_DAYS = 252  # ~1 year of sessions

# Calendar-day lookbacks. We resolve each to the last trading day on or
# before (last_date - N days), so holidays and weekends are handled.
LOOKBACKS = {
    "1w": 7,
    "4w": 28,
    "13w": 91,
    "26w": 182,
    "52w": 365,
}

DEFAULT_BATCH_SIZE = 60
DEFAULT_PAUSE = 1.0      # seconds between batches
DEFAULT_HISTORY_DAYS = 420  # ~14 months, enough for 52w metrics + buffer


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

def make_session():
    """
    curl_cffi session with browser impersonation, plus yfinance's own retry
    setting.

    Yahoo is much less aggressive about rate limiting when the TLS
    fingerprint looks like a real browser. yfinance defaults to retries=0,
    so a single transient 429 otherwise drops a whole batch; two retries
    cover the usual blip.

    Note: yfinance rejects caching sessions (requests_cache), so this must
    stay a plain session. Returns None if curl_cffi is unavailable, in which
    case yfinance falls back to its own default session.
    """
    try:
        yf.config.network.retries = 2          # yfinance >= 1.x
    except Exception:
        try:
            yf.set_config(retries=2)           # older releases
        except Exception:
            pass

    try:
        from curl_cffi import requests as cffi_requests

        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

def normalise_symbol(symbol: str) -> str:
    """
    EasyEquities uses dots for share classes (BRK.B); Yahoo uses dashes.
    """
    s = str(symbol).strip().upper()
    return s.replace(".", "-")


def load_universe(path: str | Path) -> pd.DataFrame:
    """
    Load the ticker list CSV.

    Only Symbol / Sector / Industry / Company Name are required. Any other
    columns in the export (Trade Date, Purchase Price, Quantity) are dropped,
    since they describe a holding rather than the universe.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["Symbol"] = df["Symbol"].map(normalise_symbol)

    for col in ["Sector", "Industry", "Company Name"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col] == "", col] = "Unknown"

    df = df[df["Symbol"].str.len() > 0]
    df = df.drop_duplicates(subset="Symbol").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Price download
# --------------------------------------------------------------------------

def _flatten(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    yf.download returns a MultiIndex frame for >1 ticker and a flat frame for
    exactly one. Normalise both into {symbol: OHLCV frame}.
    """
    out: dict[str, pd.DataFrame] = {}

    if raw is None or raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        available = raw.columns.get_level_values(0).unique()
        for sym in tickers:
            if sym in available:
                sub = raw[sym].dropna(how="all")
                if not sub.empty:
                    out[sym] = sub
    else:
        sub = raw.dropna(how="all")
        if not sub.empty:
            out[tickers[0]] = sub

    return out


def download_prices(
    symbols: list[str],
    days: int = DEFAULT_HISTORY_DAYS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pause: float = DEFAULT_PAUSE,
    session=None,
    progress_cb=None,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for all symbols in batches.

    Batching matters: one yf.download call with 863 symbols builds a very
    long URL and hammers Yahoo with parallel threads. Batches of ~60 with a
    short pause between them is the sweet spot in practice.

    progress_cb(done, total, label) is called after each batch so a UI can
    show progress.
    """
    session = session or make_session()
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)

    frames: dict[str, pd.DataFrame] = {}
    batches = [
        symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
    ]

    for i, batch in enumerate(batches, start=1):
        kwargs = dict(
            tickers=batch,
            start=start.date(),
            end=end.date(),
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if session is not None:
            kwargs["session"] = session

        try:
            raw = yf.download(**kwargs)
            frames.update(_flatten(raw, batch))
        except Exception as exc:  # keep going; report at the end
            print(f"[warn] batch {i}/{len(batches)} failed: {exc}")

        if progress_cb:
            progress_cb(i, len(batches), f"batch {i} of {len(batches)}")

        if i < len(batches) and pause:
            time.sleep(pause)

    return frames


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _close_asof(close: pd.Series, target: pd.Timestamp) -> float:
    """Last close on or before `target`, or NaN if we have no history that far back."""
    window = close.loc[:target]
    if window.empty:
        return np.nan
    return float(window.iloc[-1])


def compute_metrics(sym: str, ohlcv: pd.DataFrame) -> dict | None:
    """
    All price-derived metrics for one symbol. Returns None if there is not
    enough usable history.
    """
    if ohlcv is None or "Close" not in ohlcv:
        return None

    close = ohlcv["Close"].dropna()
    if len(close) < 2:
        return None

    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close = close.sort_index()

    volume = ohlcv["Volume"].reindex(close.index).fillna(0) if "Volume" in ohlcv else None

    last_date = close.index[-1]
    last = float(close.iloc[-1])

    row: dict = {
        "Symbol": sym,
        "Close": last,
        "As Of": last_date.date(),
        "Bars": int(len(close)),
    }

    # Point-in-time closes and percentage changes
    for label, days in LOOKBACKS.items():
        prior = _close_asof(close, last_date - pd.Timedelta(days=days))
        row[f"Close {label} ago"] = prior
        row[f"% {label}"] = (
            (last / prior - 1.0) * 100.0 if prior and not np.isnan(prior) and prior > 0 else np.nan
        )

    # Day change
    prev = float(close.iloc[-2])
    row["% 1d"] = (last / prev - 1.0) * 100.0 if prev > 0 else np.nan

    # Moving averages
    for n in (20, 50, 200):
        if len(close) >= n:
            sma = float(close.rolling(n).mean().iloc[-1])
            row[f"SMA{n}"] = sma
            row[f"% vs SMA{n}"] = (last / sma - 1.0) * 100.0 if sma > 0 else np.nan
        else:
            row[f"SMA{n}"] = np.nan
            row[f"% vs SMA{n}"] = np.nan

    # 52-week range (trailing 252 trading days)
    trailing = close.tail(TRADING_DAYS)
    hi, lo = float(trailing.max()), float(trailing.min())
    row["52w High"] = hi
    row["52w Low"] = lo
    row["% off 52w High"] = (last / hi - 1.0) * 100.0 if hi > 0 else np.nan
    row["% above 52w Low"] = (last / lo - 1.0) * 100.0 if lo > 0 else np.nan

    # Annualised realised volatility over the last month of trading
    rets = close.pct_change().dropna()
    row["Volatility 1m %"] = (
        float(rets.tail(21).std() * np.sqrt(252) * 100.0) if len(rets) >= 10 else np.nan
    )

    # Liquidity
    if volume is not None and len(volume) >= 5:
        row["Avg Volume 20d"] = float(volume.tail(20).mean())
        row["Avg $ Volume 20d"] = float((close * volume).tail(20).mean())
    else:
        row["Avg Volume 20d"] = np.nan
        row["Avg $ Volume 20d"] = np.nan

    return row


def build_metrics_frame(
    universe: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join computed metrics back onto the universe (Sector / Industry / Name)."""
    rows = [m for sym, f in frames.items() if (m := compute_metrics(sym, f)) is not None]
    metrics = pd.DataFrame(rows)

    if metrics.empty:
        return universe.assign(Close=np.nan)

    out = universe.merge(metrics, on="Symbol", how="left")
    out["Yahoo"] = "https://finance.yahoo.com/quote/" + out["Symbol"]
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(df: pd.DataFrame, level: str, change_col: str = "% 4w") -> pd.DataFrame:
    """
    Roll tickers up to Sector or Industry level.

    Median is the headline number rather than mean, because a single 300%
    mover distorts a sector average badly at this universe size.
    """
    if level not in ("Sector", "Industry"):
        raise ValueError("level must be 'Sector' or 'Industry'")

    valid = df[df[change_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    group_keys = [level] if level == "Sector" else ["Sector", "Industry"]
    grouped = valid.groupby(group_keys, dropna=False)

    agg = grouped.agg(
        Tickers=("Symbol", "count"),
        Median=(change_col, "median"),
        Mean=(change_col, "mean"),
        Best=(change_col, "max"),
        Worst=(change_col, "min"),
    ).reset_index()

    valid["_advancing"] = (valid[change_col] > 0).astype(int)
    advancers = (
        valid.groupby(group_keys, dropna=False)["_advancing"]
        .sum()
        .rename("Advancers")
        .reset_index()
    )
    agg = agg.merge(advancers, on=group_keys, how="left")
    agg["Decliners"] = agg["Tickers"] - agg["Advancers"]
    agg["% Advancing"] = agg["Advancers"] / agg["Tickers"] * 100.0

    best_sym = (
        valid.loc[grouped[change_col].idxmax(), group_keys + ["Symbol"]]
        .rename(columns={"Symbol": "Best Ticker"})
    )
    worst_sym = (
        valid.loc[grouped[change_col].idxmin(), group_keys + ["Symbol"]]
        .rename(columns={"Symbol": "Worst Ticker"})
    )
    agg = agg.merge(best_sym, on=group_keys, how="left")
    agg = agg.merge(worst_sym, on=group_keys, how="left")

    agg = agg.rename(
        columns={
            "Median": f"Median {change_col}",
            "Mean": f"Mean {change_col}",
            "Best": f"Best {change_col}",
            "Worst": f"Worst {change_col}",
        }
    )
    return agg.sort_values(f"Median {change_col}", ascending=False).reset_index(drop=True)
