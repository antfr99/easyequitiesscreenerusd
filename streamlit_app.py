"""
EasyEquities USD Screener
=========================

Ticker, sector and industry views over the USD universe.

Data comes from a pre-built snapshot (data/snapshot.parquet) when one is
present, and falls back to a live yfinance pull otherwise. The snapshot is
strongly preferred: it is built once a day by a GitHub Action, so the app
never has to hit Yahoo while a user is waiting, and never gets rate limited
in front of an audience.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import screener_core as core

# --------------------------------------------------------------------------
# Paths and page setup
# --------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SNAPSHOT = DATA_DIR / "snapshot.parquet"
SNAPSHOT_META = DATA_DIR / "snapshot_meta.json"

st.set_page_config(
    page_title="EasyEquities USD Screener",
    page_icon="📈",
    layout="wide",
)

CHANGE_COLS = ["% 1d", "% 1w", "% 4w", "% 13w", "% 26w", "% 52w"]


# --------------------------------------------------------------------------
# Data loading (two-tier cache: resource for the session, data for frames)
# --------------------------------------------------------------------------

@st.cache_resource
def get_session():
    return core.make_session()


@st.cache_data(show_spinner=False)
def get_snapshot(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "As Of" in df.columns:
        df["As Of"] = pd.to_datetime(df["As Of"]).dt.date
    return df


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_live(csv_path: str, mtime: float, days: int) -> tuple[pd.DataFrame, list[str]]:
    """Live pull. Cached for an hour so a rerun does not re-download."""
    universe, skipped = core.load_universe(csv_path)
    symbols = universe["Symbol"].tolist()

    bar = st.progress(0.0, text="Contacting Yahoo Finance…")

    def on_progress(done, total, label):
        bar.progress(done / total, text=f"Downloading prices — {label}")

    frames = core.download_prices(
        symbols,
        days=days,
        session=get_session(),
        progress_cb=on_progress,
    )
    bar.empty()
    return core.build_metrics_frame(universe, frames), skipped


def load_data(csv_path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Returns (frame, source_label, csv_warnings).

    Order of preference:
      1. A pre-built snapshot (data/snapshot.parquet) — instant, no Yahoo call.
      2. A live pull the user has already triggered this session (kept in
         st.session_state so it survives reruns from filtering).
      3. The bare universe with empty price columns — instant, so the page
         always renders. Prices then load only when the user clicks the
         button, never automatically at page load.

    This matters most on Streamlit Cloud: an automatic 863-ticker download
    on every visit sits on a shared IP that Yahoo throttles, so the page
    would hang before rendering. On-demand loading keeps first paint instant.
    """
    if SNAPSHOT.exists():
        df = get_snapshot(str(SNAPSHOT), SNAPSHOT.stat().st_mtime)
        stamp = "unknown"
        csv_warnings: list[str] = []
        if SNAPSHOT_META.exists():
            try:
                meta = json.loads(SNAPSHOT_META.read_text())
                stamp = meta.get("built_at", "unknown")
                csv_warnings = meta.get("csv_warnings", [])
            except Exception:
                pass
        return df, f"Snapshot built {stamp}", csv_warnings

    # A live pull the user triggered earlier this session.
    if "live_df" in st.session_state:
        return (
            st.session_state["live_df"],
            st.session_state.get("live_source", "Live pull"),
            st.session_state.get("live_warnings", []),
        )

    # Default: universe only, no prices, no network call.
    universe, skipped = core.load_universe(str(csv_path))
    for col in core.EMPTY_METRIC_COLUMNS:
        universe[col] = np.nan
    universe["Yahoo"] = "https://finance.yahoo.com/quote/" + universe["Symbol"]
    return universe, "No snapshot yet — prices not loaded", skipped


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def pct_column(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="%.2f%%")


def price_column(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="$%.2f")


def ticker_column_config() -> dict:
    cfg = {
        "Yahoo": st.column_config.LinkColumn("Chart", display_text="open"),
        "Close": price_column("Close"),
        "Close 4w ago": price_column("Close 4w ago"),
        "52w High": price_column("52w High"),
        "52w Low": price_column("52w Low"),
        "SMA20": price_column("SMA20"),
        "SMA50": price_column("SMA50"),
        "SMA200": price_column("SMA200"),
        "Avg Volume 20d": st.column_config.NumberColumn("Avg Vol 20d", format="%.0f"),
        "Avg $ Volume 20d": st.column_config.NumberColumn("Avg $ Vol 20d", format="$%.0f"),
        "Volatility 1m %": pct_column("Vol 1m (ann.)"),
    }
    for c in CHANGE_COLS + ["% vs SMA20", "% vs SMA50", "% vs SMA200",
                            "% off 52w High", "% above 52w Low"]:
        cfg[c] = pct_column(c)
    return cfg


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------

def sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    st.sidebar.header("Filters")

    available_change_cols = [c for c in CHANGE_COLS if c in df.columns]
    if not available_change_cols:
        st.sidebar.warning("No price data loaded yet — filters are limited.")
        search = st.sidebar.text_input("Search ticker or company", placeholder="e.g. AEHR")
        out = df.copy()
        if search.strip():
            q = search.strip().lower()
            out = out[
                out["Symbol"].str.lower().str.contains(q, na=False)
                | out["Company Name"].str.lower().str.contains(q, na=False)
            ]
        return out, None

    change_col = st.sidebar.selectbox(
        "Performance window",
        available_change_cols,
        index=available_change_cols.index("% 4w") if "% 4w" in available_change_cols else 0,
        help="Drives the sliders below and the sector / industry rankings.",
    )

    search = st.sidebar.text_input("Search ticker or company", placeholder="e.g. AEHR")

    sectors = sorted(df["Sector"].dropna().unique())
    chosen_sectors = st.sidebar.multiselect("Sector", sectors, default=[])

    scoped = df[df["Sector"].isin(chosen_sectors)] if chosen_sectors else df
    industries = sorted(scoped["Industry"].dropna().unique())
    chosen_industries = st.sidebar.multiselect("Industry", industries, default=[])

    st.sidebar.divider()

    # Performance range
    series = df[change_col].replace([np.inf, -np.inf], np.nan).dropna()
    if not series.empty:
        lo = float(np.floor(max(series.min(), -100)))
        hi = float(np.ceil(min(series.max(), 500)))
        perf_range = st.sidebar.slider(
            f"{change_col} range",
            min_value=lo,
            max_value=hi,
            value=(lo, hi),
            step=1.0,
        )
    else:
        perf_range = None

    # Price range
    prices = df["Close"].dropna()
    if not prices.empty:
        price_range = st.sidebar.slider(
            "Close price ($)",
            min_value=0.0,
            max_value=float(np.ceil(prices.max())),
            value=(0.0, float(np.ceil(prices.max()))),
        )
    else:
        price_range = None

    min_dollar_vol = st.sidebar.number_input(
        "Min avg $ volume (20d)",
        min_value=0,
        value=0,
        step=100_000,
        help="Liquidity floor. 1,000,000 filters out most of the untradeable tail.",
    )

    st.sidebar.divider()
    trend = st.sidebar.radio(
        "Trend",
        ["Any", "Above SMA50", "Below SMA50", "Above SMA200", "Below SMA200"],
        index=0,
    )
    near_high = st.sidebar.checkbox("Within 10% of 52w high")
    hide_missing = st.sidebar.checkbox("Hide tickers with no price data", value=True)
    hide_stale = (
        st.sidebar.checkbox(
            "Hide stale rows",
            value=False,
            help="Rows carried forward from an earlier snapshot because Yahoo "
                 "throttled the last refresh.",
        )
        if "Stale" in df.columns
        else False
    )

    # ----- apply -----
    out = df.copy()

    if search.strip():
        q = search.strip().lower()
        out = out[
            out["Symbol"].str.lower().str.contains(q, na=False)
            | out["Company Name"].str.lower().str.contains(q, na=False)
        ]
    if chosen_sectors:
        out = out[out["Sector"].isin(chosen_sectors)]
    if chosen_industries:
        out = out[out["Industry"].isin(chosen_industries)]
    if perf_range:
        out = out[out[change_col].between(*perf_range) | out[change_col].isna()]
    if price_range:
        out = out[out["Close"].between(*price_range) | out["Close"].isna()]
    if min_dollar_vol > 0 and "Avg $ Volume 20d" in out.columns:
        out = out[out["Avg $ Volume 20d"] >= min_dollar_vol]

    if trend == "Above SMA50":
        out = out[out["% vs SMA50"] > 0]
    elif trend == "Below SMA50":
        out = out[out["% vs SMA50"] < 0]
    elif trend == "Above SMA200":
        out = out[out["% vs SMA200"] > 0]
    elif trend == "Below SMA200":
        out = out[out["% vs SMA200"] < 0]

    if near_high and "% off 52w High" in out.columns:
        out = out[out["% off 52w High"] >= -10]

    if hide_missing:
        out = out[out["Close"].notna()]
    if hide_stale:
        out = out[~out["Stale"].fillna(False)]

    return out, change_col


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------

def ticker_view(df: pd.DataFrame, change_col: str | None) -> None:
    st.subheader("Tickers")

    display_cols = [
        "Symbol", "Company Name", "Sector", "Industry",
        "Close", "Close 4w ago", "% 4w",
        "% 1d", "% 1w", "% 13w", "% 26w", "% 52w",
        "% vs SMA50", "% vs SMA200", "% off 52w High",
        "Volatility 1m %", "Avg $ Volume 20d", "As Of", "Yahoo",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    if change_col and change_col in df.columns:
        view = df[display_cols].sort_values(change_col, ascending=False, na_position="last")
    else:
        view = df[display_cols].sort_values("Symbol")

    st.dataframe(
        view,
        column_config=ticker_column_config(),
        hide_index=True,
        width="stretch",
        height=560,
    )

    st.download_button(
        "Download these tickers (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="screener_tickers.csv",
        mime="text/csv",
    )

    if len(view) and change_col and change_col in view.columns:
        left, right = st.columns(2)
        top = view.nlargest(15, change_col)[["Symbol", change_col]]
        bottom = view.nsmallest(15, change_col)[["Symbol", change_col]]
        with left:
            st.caption(f"Top 15 by {change_col}")
            st.bar_chart(top.set_index("Symbol"), horizontal=True)
        with right:
            st.caption(f"Bottom 15 by {change_col}")
            st.bar_chart(bottom.set_index("Symbol"), horizontal=True)


def group_view(df: pd.DataFrame, level: str, change_col: str | None) -> None:
    st.subheader(level)

    if not change_col:
        st.info("No price data loaded yet, so there's nothing to rank by.")
        return

    agg = core.aggregate(df, level, change_col)
    if agg.empty:
        st.warning("Nothing to aggregate with the current filters.")
        return

    median_col = f"Median {change_col}"
    cfg = {
        c: pct_column(c)
        for c in agg.columns
        if c.startswith(("Median", "Mean", "Best %", "Worst %", "% Advancing"))
    }
    cfg["Tickers"] = st.column_config.NumberColumn("Tickers", format="%d")

    st.dataframe(
        agg,
        column_config=cfg,
        hide_index=True,
        width="stretch",
        height=480,
    )

    chart = agg.set_index(level if level == "Sector" else "Industry")[median_col]
    st.caption(f"{median_col} by {level.lower()}")
    st.bar_chart(chart, horizontal=True)

    st.download_button(
        f"Download {level.lower()} summary (CSV)",
        agg.to_csv(index=False).encode("utf-8"),
        file_name=f"screener_{level.lower()}.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    st.title("📈 EasyEquities USD Screener")

    csv_path, notes = core.find_universe_csv(ROOT, DATA_DIR)
    if csv_path is None:
        st.error(
            "Could not find the universe CSV. Expected a file with Symbol, "
            "Sector and Industry columns in data/ or the repo root."
        )
        if notes:
            with st.expander("Why", expanded=True):
                for n in notes:
                    st.write("•", n)
        st.stop()

    df, source, csv_warnings = load_data(csv_path)

    if "Close" not in df.columns:
        st.error("The loaded data has no price columns. Rebuild the snapshot.")
        st.stop()

    # When there is no snapshot and no session pull yet, prices are all NaN.
    # Offer to load them on demand instead of blocking the page at startup.
    prices_loaded = df["Close"].notna().any()
    no_snapshot = not SNAPSHOT.exists()

    if no_snapshot:
        loaded_sectors = st.session_state.get("loaded_sectors", [])
        universe_all, _ = core.load_universe(str(csv_path))
        all_sectors = sorted(universe_all["Sector"].dropna().unique())
        remaining = [s for s in all_sectors if s not in loaded_sectors]

        with st.container():
            st.warning(
                "No daily snapshot is committed yet, so prices load on demand. "
                "Pick a sector and load it — each is a small, fast pull that's "
                "far less likely to be throttled than the whole universe at once. "
                "Load as many sectors as you like; they accumulate.",
                icon="⏳",
            )
            c1, c2 = st.columns([2, 1])
            with c1:
                if remaining:
                    sector_to_load = st.selectbox(
                        "Sector to load",
                        remaining,
                        help=f"{len(loaded_sectors)} of {len(all_sectors)} sectors loaded so far.",
                    )
                else:
                    sector_to_load = None
                    st.success("All sectors loaded.")
            with c2:
                st.write("")
                st.write("")
                load_all = st.button("Load ALL remaining", help="Slower; may be throttled.")

            if loaded_sectors:
                st.caption("Loaded: " + ", ".join(loaded_sectors))

            targets: list[str] = []
            if sector_to_load and st.button(f"Load {sector_to_load}", type="primary"):
                targets = [sector_to_load]
            elif load_all and remaining:
                targets = remaining

            if targets:
                base = df.copy()
                subset = universe_all[universe_all["Sector"].isin(targets)]
                with st.spinner(f"Downloading {len(subset)} tickers in {', '.join(targets)}…"):
                    frames = core.download_prices(
                        subset["Symbol"].tolist(),
                        days=core.DEFAULT_HISTORY_DAYS,
                        session=get_session(),
                    )
                    merged = core.merge_metrics(base, universe_all, frames)
                st.session_state["live_df"] = merged
                st.session_state["loaded_sectors"] = loaded_sectors + targets
                st.session_state["live_source"] = (
                    f"Live pull {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} — "
                    f"{len(loaded_sectors) + len(targets)} sector(s)"
                )
                st.rerun()

    if csv_warnings:
        with st.expander(f"⚠️ {len(csv_warnings)} row(s) in the CSV needed attention", expanded=False):
            st.caption(
                "Usually an unquoted comma inside Company Name (e.g. \"Smith, Jones & Co\") "
                "throws off the column count for that line. Wrap the name in quotes in the "
                "CSV to fix it permanently."
            )
            for w in csv_warnings:
                st.write("•", w)

    filtered, change_col = sidebar_filters(df)

    covered = int(df["Close"].notna().sum())
    cols = st.columns(4)
    cols[0].metric("Universe", f"{len(df):,}")
    cols[1].metric("With price data", f"{covered:,}")
    cols[2].metric("Matching filters", f"{len(filtered):,}")
    if change_col and change_col in filtered.columns and filtered[change_col].notna().any():
        cols[3].metric(f"Median {change_col}", f"{filtered[change_col].median():.2f}%")

    st.caption(source)

    if covered < len(df):
        missing = len(df) - covered
        st.caption(
            f"{missing} symbol(s) returned no data — usually delistings, "
            "ticker changes, or share classes Yahoo spells differently."
        )

    tab_t, tab_s, tab_i = st.tabs(["Tickers", "Sectors", "Industries"])
    with tab_t:
        ticker_view(filtered, change_col)
    with tab_s:
        group_view(filtered, "Sector", change_col)
    with tab_i:
        group_view(filtered, "Industry", change_col)

    with st.sidebar:
        st.divider()
        if "live_df" in st.session_state:
            if st.button("Clear loaded prices"):
                for k in ("live_df", "live_source", "live_warnings", "loaded_sectors"):
                    st.session_state.pop(k, None)
                st.rerun()
        if st.button("Clear cache and reload"):
            st.cache_data.clear()
            for k in ("live_df", "live_source", "live_warnings", "loaded_sectors"):
                st.session_state.pop(k, None)
            st.rerun()


if __name__ == "__main__":
    main()
