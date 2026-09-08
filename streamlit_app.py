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
    page_title="EasyEquities USD Equity Screener",
    page_icon="📈",
    layout="wide",
)

CHANGE_COLS = ["% 1d", "% 1w", "% 4w", "% 13w", "% 26w", "% 52w"]

# Force on-demand mode: ignore any committed snapshot and never auto-load.
# The app starts with the bare universe (no prices, no network call); prices
# load only when the user clicks the sidebar run button. Set to False to go
# back to preferring data/snapshot.parquet.
USE_ON_DEMAND_ONLY = True


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


def load_data(csv_path: Path) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    """
    Returns (frame, source_label, errors, notes).

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
    if not USE_ON_DEMAND_ONLY and SNAPSHOT.exists():
        df = get_snapshot(str(SNAPSHOT), SNAPSHOT.stat().st_mtime)
        stamp = "unknown"
        errors: list[str] = []
        notes: list[str] = []
        if SNAPSHOT_META.exists():
            try:
                meta = json.loads(SNAPSHOT_META.read_text())
                stamp = meta.get("built_at", "unknown")
                errors = meta.get("csv_errors", meta.get("csv_warnings", []))
                notes = meta.get("csv_notes", [])
            except Exception:
                pass
        return df, f"Snapshot built {stamp}", errors, notes

    # A live pull the user triggered earlier this session.
    if "live_df" in st.session_state:
        return (
            st.session_state["live_df"],
            st.session_state.get("live_source", "Live pull"),
            st.session_state.get("live_errors", []),
            st.session_state.get("live_notes", []),
        )

    # Default: universe only, no prices, no network call.
    universe, errors, notes = core.load_universe(str(csv_path))
    for col in core.EMPTY_METRIC_COLUMNS:
        universe[col] = np.nan
    universe["Yahoo"] = "https://finance.yahoo.com/quote/" + universe["Symbol"]
    return universe, "No snapshot yet — prices not loaded", errors, notes


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

def _load_scope(csv_path: Path, scope: pd.DataFrame, base: pd.DataFrame) -> None:
    """
    Download prices for exactly the tickers in `scope` and merge into base.

    `scope` is whatever the sidebar's current Sector/Industry selection
    resolves to — it may be a whole sector or a single industry within it.
    Loading only ever fetches this exact set, never a wider one; which
    tickers have been fetched is tracked at the symbol level (not by
    sector), so picking a narrower Industry after a broader Sector never
    re-downloads more than the gap.
    """
    universe_all, _errors, _notes = core.load_universe(str(csv_path))
    symbols = scope["Symbol"].tolist()
    with st.spinner(f"Downloading {len(symbols)} ticker(s)…"):
        frames = core.download_prices(
            symbols,
            days=core.DEFAULT_HISTORY_DAYS,
            session=get_session(),
        )
        merged = core.merge_metrics(base, universe_all, frames)
    loaded = set(st.session_state.get("loaded_symbols", []))
    st.session_state["live_df"] = merged
    st.session_state["loaded_symbols"] = sorted(loaded | set(symbols))
    st.session_state["live_errors"] = _errors
    st.session_state["live_notes"] = _notes
    st.session_state["live_source"] = (
        f"Live pull {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} — "
        f"{len(st.session_state['loaded_symbols'])} ticker(s) loaded"
    )
    st.rerun()


_FILTER_BUTTON_CSS = """
<style>
div[class*="st-key-filter_scope_btn"] button {
    background-color: #cfe8fb;
    color: #0b4f79;
    border: 1px solid #9cc9e8;
}
div[class*="st-key-filter_scope_btn"] button:hover {
    background-color: #b8ddf6;
    color: #0b4f79;
    border-color: #7fb3d5;
}
div[class*="st-key-filter_scope_btn"] button:active {
    background-color: #a5d2f2;
}
</style>
"""


def sidebar_filters(
    df: pd.DataFrame,
    csv_path: Path | None = None,
    no_snapshot: bool = False,
) -> tuple[pd.DataFrame, str | None]:
    st.sidebar.header("Filters")

    search = st.sidebar.text_input("Search ticker or company", placeholder="e.g. AEHR")

    sectors = sorted(df["Sector"].dropna().unique())
    chosen_sectors = st.sidebar.multiselect("Sector", sectors, default=[])

    # Industry list is scoped to the chosen sector(s): pick Financials and
    # only Financials industries appear. With no sector chosen, all show.
    scoped = df[df["Sector"].isin(chosen_sectors)] if chosen_sectors else df
    industries = sorted(scoped["Industry"].dropna().unique())
    chosen_industries = st.sidebar.multiselect("Industry", industries, default=[])

    # Loading lives below both pickers and respects both: selecting an
    # Industry narrows the download to that industry, not the whole sector.
    if no_snapshot and csv_path is not None:
        loaded_symbols = set(st.session_state.get("loaded_symbols", []))
        current_scope = df
        if chosen_sectors:
            current_scope = current_scope[current_scope["Sector"].isin(chosen_sectors)]
        if chosen_industries:
            current_scope = current_scope[current_scope["Industry"].isin(chosen_industries)]

        if chosen_sectors or chosen_industries:
            to_load = current_scope[~current_scope["Symbol"].isin(loaded_symbols)]
            st.sidebar.markdown(_FILTER_BUTTON_CSS, unsafe_allow_html=True)
            if len(to_load) > 0:
                if st.sidebar.button(
                    "Filter Sector/Industries",
                    key="filter_scope_btn",
                    use_container_width=True,
                ):
                    _load_scope(csv_path, to_load, df)
                st.sidebar.caption(
                    f"{len(to_load)} of {len(current_scope)} ticker(s) in this "
                    "selection still need prices."
                )
            else:
                st.sidebar.success(f"All {len(current_scope)} ticker(s) in this selection are loaded.")
        elif loaded_symbols:
            st.sidebar.caption(f"{len(loaded_symbols)} ticker(s) loaded so far.")
        else:
            st.sidebar.caption("Select a sector or industry above, then load its prices.")

    available_change_cols = [c for c in CHANGE_COLS if c in df.columns]
    has_prices = bool(available_change_cols) and df["Close"].notna().any()

    if not has_prices:
        # No price data yet — sector/industry/search still work; skip the
        # performance and price filters that would have nothing to act on.
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
        return out, None

    st.sidebar.divider()

    change_col = st.sidebar.selectbox(
        "Performance window",
        available_change_cols,
        index=available_change_cols.index("% 4w") if "% 4w" in available_change_cols else 0,
        help="Drives the sliders below and the sector / industry rankings.",
    )

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


def no_data_view(df: pd.DataFrame) -> None:
    st.subheader("No Data")

    cols = [c for c in ["Symbol", "Company Name", "Sector", "Industry", "Yahoo"] if c in df.columns]
    missing = df.loc[df["Close"].isna(), cols].sort_values("Symbol")

    st.caption(
        f"{len(missing)} ticker(s) returned no price data — usually delistings, "
        "ticker changes, or share classes Yahoo spells differently."
    )

    if missing.empty:
        st.success("Every ticker in the universe has price data.")
        return

    col_cfg = {}
    if "Yahoo" in missing.columns:
        col_cfg["Yahoo"] = st.column_config.LinkColumn("Chart", display_text="open")

    st.dataframe(
        missing,
        column_config=col_cfg,
        hide_index=True,
        width="stretch",
        height=560,
    )

    st.download_button(
        "Download tickers with no data (CSV)",
        missing.to_csv(index=False).encode("utf-8"),
        file_name="screener_no_data.csv",
        mime="text/csv",
    )


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
    st.title("📈 EasyEquities USD Equity Screener")

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

    df, source, csv_errors, csv_notes = load_data(csv_path)

    if "Close" not in df.columns:
        st.error("The loaded data has no price columns. Rebuild the snapshot.")
        st.stop()

    # When there is no snapshot, prices load on demand via the sidebar
    # Sector filter (see sidebar_filters). Show a short banner explaining that.
    no_snapshot = USE_ON_DEMAND_ONLY or not SNAPSHOT.exists()

    if no_snapshot:
        loaded_symbols = st.session_state.get("loaded_symbols", [])
        if not loaded_symbols:
            st.info(
                "Prices load on demand. Pick a sector and/or industry in the left "
                "sidebar, then click the load button — selections accumulate. "
                "This is a personal hobby project and is **not affiliated with, "
                "endorsed by, or connected to EasyEquities** in any way.",
                icon="⏳",
            )
        else:
            st.caption(
                "Not affiliated with, endorsed by, or connected to EasyEquities — "
                "a personal hobby project. - Ticker List last updated 8th September 2026"
            )

    # Only genuinely unparseable rows get a warning the user should act on.
    if csv_errors:
        with st.expander(f"⚠️ {len(csv_errors)} row(s) couldn't be read", expanded=False):
            st.caption(
                "These rows had a column count we couldn't recover — usually more "
                "than one stray comma. Check them in the CSV; everything else loaded."
            )
            for w in csv_errors:
                st.write("•", w)

    # Duplicates and other housekeeping are informational, not problems.
    if csv_notes:
        st.caption(" · ".join(csv_notes))

    filtered, change_col = sidebar_filters(df, csv_path, no_snapshot)

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

    tab_t, tab_s, tab_i, tab_nd = st.tabs(["Tickers", "Sectors", "Industries", "No Data"])
    with tab_t:
        ticker_view(filtered, change_col)
    with tab_s:
        group_view(filtered, "Sector", change_col)
    with tab_i:
        group_view(filtered, "Industry", change_col)
    with tab_nd:
        no_data_view(df)

    with st.sidebar:
        st.divider()
        if "live_df" in st.session_state:
            if st.button("Clear loaded prices"):
                for k in ("live_df", "live_source", "live_errors", "live_notes", "loaded_symbols"):
                    st.session_state.pop(k, None)
                st.rerun()
        if st.button("Clear cache and reload"):
            st.cache_data.clear()
            for k in ("live_df", "live_source", "live_errors", "live_notes", "loaded_symbols"):
                st.session_state.pop(k, None)
            st.rerun()


if __name__ == "__main__":
    main()
