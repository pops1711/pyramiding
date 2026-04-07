"""
Pyramiding Portfolio Dashboard — Fixed
=======================================
Bugs fixed in this version:
  1. Near SL blank   → SL column lookup made robust (fuzzy match)
  2. Exited Rs 0     → realized P&L fallback used scalar 0 when column not found;
                        now uses file's 'Profit & Loss' as proper fallback
  3. Top 10% invisible text → bg:#14532d + text:#86efac = low contrast; now white text
  4. Session cache not cleared on new upload → clear on file name change
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import yfinance as yf

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# ─────────────────────── PAGE CONFIG ─────────────────────────────
st.set_page_config(page_title="Pyramiding Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

PYRAMID_STEP = 1.05
STEP_TOL     = 0.025   # ±2.5% for level matching
DUP_TOL      = 0.010   # <1% = duplicate

PLOT_LAYOUT = dict(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                   font_color="#e2e8f0", margin=dict(l=10, r=10, t=40, b=10))

# ─────────────────────── CSS ──────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{
    background:#1a1f2e;border-radius:10px;
    padding:14px 18px;border-left:4px solid #6366f1;}
[data-testid="stMetricLabel"]{color:#94a3b8;font-size:.78rem;}
[data-testid="stMetricValue"]{color:#f1f5f9;font-size:1.3rem;font-weight:700;}
.stTabs [data-baseweb="tab"]{border-radius:6px 6px 0 0;font-weight:600;}
h2,h3{color:#e2e8f0!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────── PRICE HELPERS ───────────────────────────
@st.cache_data(ttl=60)
def fetch_stock_stats(ticker: str, start_date=None) -> dict:
    for sym in [f"{ticker.upper()}.NS", ticker.upper()]:
        try:
            if start_date and pd.notna(start_date):
                h = yf.download(sym, start=start_date.strftime("%Y-%m-%d"), progress=False)
            else:
                h = yf.download(sym, period="1y", progress=False)
            if not h.empty:
                cmp = float(h["Close"].squeeze().iloc[-1])
                hh  = float(h["High"].squeeze().max())
                return {"cmp": cmp, "hh": hh}
        except Exception:
            pass
    return {"cmp": None, "hh": None}

@st.cache_data(ttl=3600)
def prev_close(ticker: str):
    for sym in [f"{ticker.upper()}.NS", ticker.upper()]:
        try:
            h = yf.Ticker(sym).history(period="3d")
            if len(h) >= 2:
                return float(h["Close"].iloc[-2])
            if len(h) == 1:
                return float(h["Close"].iloc[0])
        except Exception:
            pass
    return None

# ─────────────────────── COLUMN FINDER (robust) ──────────────────
def find_col(df: pd.DataFrame, candidates: list):
    """
    Return the first column name in df.columns that case-insensitively
    matches any candidate string (stripped, no spaces/slashes).
    Returns None if not found.
    """
    def norm(s): return str(s).strip().lower().replace(" ","").replace("/","").replace("&","")
    normed = {norm(c): c for c in df.columns}
    for cand in candidates:
        hit = normed.get(norm(cand))
        if hit:
            return hit
    return None

# ─────────────────────── FILE PARSING ────────────────────────────
def load_raw(uploaded) -> pd.DataFrame:
    name = uploaded if isinstance(uploaded, str) else uploaded.name
    return pd.read_csv(uploaded) if name.endswith(".csv") \
           else pd.read_excel(uploaded, sheet_name=0)

def parse_sheet(raw: pd.DataFrame):
    """Split one sheet → (open_df, exited_df)."""
    df = raw.copy()

    # Drop Excel auto-generated date column names
    df.columns = [
        "CMP_DATE" if (hasattr(c, "strftime") or
                       (isinstance(c, str) and len(c) >= 4 and c[:4].isdigit()))
        else str(c).strip()
        for c in df.columns
    ]

    # Numeric coercion — match by fuzzy find so slight name variants still work
    NUMERIC_CANDIDATES = [
        "QTY", "ENTRY PRICE", "Profit & Loss", "UNREALIESE P&l",
        "Investment Value", "CMP", "HighestHigh", "SL",
        "% Change", "% from Highest high to cmp",
        "CLOSE / Exit price", "Value", "CLOSE/Exit price", "Exit price", "Sell Price",
    ]
    for cand in NUMERIC_CANDIDATES:
        col = find_col(df, [cand])
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col_name in ["ENTRY DATE", "EXIT DATE"]:
        c = find_col(df, [col_name])
        if c:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if c != col_name:           # rename to standard name
                df.rename(columns={c: col_name}, inplace=True)

    # Fill null Mcap
    mcap_col = find_col(df, ["Mcap", "Market Cap", "MarketCap"])
    if mcap_col:
        df[mcap_col] = df[mcap_col].fillna("Unlisted")
        if mcap_col != "Mcap":
            df.rename(columns={mcap_col: "Mcap"}, inplace=True)

    # Identify Buy column
    buy_col = find_col(df, ["Buy", "Buy/Exit", "BuyExit", "Position", "BuySell", "Type"])
    exit_date_col = "EXIT DATE"   # already normalised above

    if buy_col:
        open_df   = df[df[buy_col].astype(str).str.upper().str.strip() == "BUY"].copy()
        mask_exit = df[buy_col].astype(str).str.upper().str.strip().isin(["EXIT", "CLOSE", "SELL"])
        if exit_date_col in df.columns:
            mask_exit = mask_exit | df[exit_date_col].notna()
        exited_df = df[mask_exit].copy()
    else:
        open_df   = df[df.get(exit_date_col, pd.Series(dtype=object)).isna()].copy()
        exited_df = df[df[exit_date_col].notna()].copy() \
                    if exit_date_col in df.columns else pd.DataFrame()

    scrip_col = find_col(df, ["SCRIP", "Symbol", "Ticker", "Stock"])
    qty_col   = find_col(df, ["QTY", "Qty", "Quantity"])
    ep_col    = find_col(df, ["ENTRY PRICE", "Entry Price", "Buy Price"])

    if scrip_col and scrip_col != "SCRIP":
        open_df.rename(columns={scrip_col: "SCRIP"}, inplace=True)
        if not exited_df.empty:
            exited_df.rename(columns={scrip_col: "SCRIP"}, inplace=True)

    open_df   = open_df.dropna(subset=["SCRIP"])
    exited_df = exited_df.dropna(subset=["SCRIP"]) if not exited_df.empty else pd.DataFrame()
    return open_df, exited_df

# ─────────────────────── PORTFOLIO BUILDER ───────────────────────
def build_portfolio(open_df: pd.DataFrame, use_live: bool) -> pd.DataFrame:
    scrips = open_df["SCRIP"].dropna().unique().tolist()

    # Resolve column names robustly once
    col_cmp  = find_col(open_df, ["CMP", "Current Price", "LTP"])
    col_hh   = find_col(open_df, ["HighestHigh", "Highest High", "52W High"])
    col_sl   = find_col(open_df, ["SL", "Stop Loss", "StopLoss"])
    col_pct  = find_col(open_df, ["% from Highest high to cmp",
                                   "% from HH", "Pct from HH"])
    col_qty  = find_col(open_df, ["QTY", "Qty", "Quantity"])
    col_ep   = find_col(open_df, ["ENTRY PRICE", "Entry Price", "Buy Price"])
    col_mcap = find_col(open_df, ["Mcap", "Market Cap"])
    col_days = find_col(open_df, ["Days After Investing", "Days Held", "Days After Invest"])
    col_date = find_col(open_df, ["ENTRY DATE", "Entry Date", "Date"])
    col_pnl  = find_col(open_df, ["Profit & Loss", "PnL", "P&L"])

    # Find first date per scrip for accurate HH calculation if using live data
    scrip_first_dates = {}
    if use_live and col_date:
        for s in scrips:
            scrip_first_dates[s] = open_df[open_df["SCRIP"] == s][col_date].min()

    # Live price & HH map
    price_map = {}
    hh_map    = {}
    if use_live:
        prog = st.progress(0, text="Fetching live prices & highs…")
        for i, s in enumerate(scrips):
            stats = fetch_stock_stats(s, scrip_first_dates.get(s))
            price_map[s] = stats.get("cmp")
            hh_map[s]    = stats.get("hh")
            prog.progress((i + 1) / len(scrips), text=f"Fetching {s}…")
        prog.empty()
    else:
        if col_cmp:
            for s in scrips:
                vals = open_df[open_df["SCRIP"] == s][col_cmp].dropna()
                price_map[s] = float(vals.iloc[-1]) if not vals.empty else None

    rows = []
    for scrip in scrips:
        g = open_df[open_df["SCRIP"] == scrip].copy()
        if col_ep:
            g = g.sort_values(col_ep)
        else:
            continue

        qty_total = g[col_qty].sum() if col_qty else 0
        inv_total = (g[col_qty] * g[col_ep]).sum() if col_qty and col_ep else 0
        avg_price = inv_total / qty_total if qty_total else 0

        cmp = price_map.get(scrip)
        cmp_live = bool(cmp and not np.isnan(float(cmp)))
        if not cmp_live:
            cmp = avg_price

        unrealized = (cmp - avg_price) * qty_total
        ret_pct    = unrealized / inv_total * 100 if inv_total else 0

        def last_val(col):
            if not col or col not in g.columns:
                return np.nan
            v = g[col].dropna()
            return float(v.iloc[-1]) if not v.empty else np.nan

        hh_local     = last_val(col_hh)
        sl_local     = last_val(col_sl)
        pct_hh_local = last_val(col_pct)
        first_date   = g[col_date].min() if col_date else pd.NaT

        # Apply Live or computed fallback
        hh = hh_map.get(scrip) if use_live and hh_map.get(scrip) else hh_local
        
        # Calculate Auto SL: 40% below highest high (meaning SL = 60% of HH)
        if pd.notna(hh) and (pd.isna(sl_local) or use_live):
            sl = hh * 0.60
        else:
            sl = sl_local

        if pd.notna(hh) and cmp:
            pct_hh = (cmp - hh) / hh
        else:
            pct_hh = pct_hh_local

        mcap   = g[col_mcap].dropna().iloc[0] if col_mcap and not g[col_mcap].dropna().empty else "Unlisted"
        days_h = g[col_days].mean()            if col_days else np.nan

        ep          = sorted(g[col_ep].tolist())
        first_entry = ep[0]
        last_entry  = ep[-1]
        next_buy    = round(last_entry * PYRAMID_STEP, 2)
        gap_next    = (next_buy - cmp) / cmp * 100 if cmp else np.nan

        rows.append(dict(
            SCRIP=scrip, Mcap=mcap,
            N_Entries=len(g), Qty=qty_total,
            Avg_Price=avg_price, Inv=inv_total,
            CMP=cmp, CMP_Live=cmp_live,
            Cur_Val=cmp * qty_total,
            Unrealized=unrealized, Ret_Pct=ret_pct,
            HH=hh, SL=sl, Pct_HH=pct_hh,
            Days_Held=days_h, Entry_Prices=ep,
            First_Entry=first_entry, Last_Entry=last_entry,
            Next_Buy=next_buy, Gap_Next=gap_next,
            First_Date=first_date,
        ))
    return pd.DataFrame(rows)

# ─────────────────────── REALIZED P&L ────────────────────────────
def realized_pnl(exited_df: pd.DataFrame):
    """
    Compute realized P&L.
    Priority: (exit_price - entry_price) × qty
    Fallback:  use file's 'Profit & Loss' column directly (already computed correctly there)
    """
    if exited_df.empty:
        return 0.0, pd.DataFrame()

    ex = exited_df.copy()

    col_exit_px = find_col(ex, ["CLOSE / Exit price", "CLOSE/Exit price",
                                 "Exit Price", "Sell Price"])
    col_ep      = find_col(ex, ["ENTRY PRICE", "Entry Price", "Buy Price"])
    col_qty     = find_col(ex, ["QTY", "Qty", "Quantity"])
    col_pnl     = find_col(ex, ["Profit & Loss", "PnL", "P&L",
                                  "UNREALIESE P&l", "Profit/Loss"])

    if col_exit_px and col_ep and col_qty:
        ex["Row_PnL"] = (ex[col_exit_px] - ex[col_ep]) * ex[col_qty]
    elif col_pnl:
        # Fallback: use file's P&L column directly
        ex["Row_PnL"] = pd.to_numeric(ex[col_pnl], errors="coerce").fillna(0)
    else:
        return 0.0, pd.DataFrame()

    total = float(ex["Row_PnL"].sum())

    agg_dict = dict(
        Entries=("SCRIP", "count"),
        Realized_PnL=("Row_PnL", "sum"),
        Exit_Date=("EXIT DATE", "max") if "EXIT DATE" in ex.columns else ("SCRIP", "count"),
    )
    if col_qty:
        agg_dict["Total_Qty"] = (col_qty, "sum")
    if col_ep:
        agg_dict["Avg_Entry"] = (col_ep, "mean")
    if col_exit_px:
        agg_dict["Exit_Price"] = (col_exit_px, "mean")

    by_scrip = ex.groupby("SCRIP").agg(**agg_dict).reset_index()
    return total, by_scrip

# ─────────────────────── ANALYSIS ────────────────────────────────
def find_missing_levels(port: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in port.iterrows():
        ep = r["Entry_Prices"]
        if len(ep) < 2:
            continue
        exp = [ep[0]]
        while exp[-1] < max(ep) * 1.06:
            exp.append(round(exp[-1] * PYRAMID_STEP, 2))
        for i, e in enumerate(exp[1:], 1):
            if e > max(ep) * (1 + STEP_TOL):
                break
            if not any(abs(p - e) / e < STEP_TOL for p in ep):
                rows.append({"SCRIP": r["SCRIP"], "Level": i + 1,
                             "Expected (Rs)": e, "Prev Level (Rs)": exp[i - 1],
                             "Gap %": f"{(e/exp[i-1]-1)*100:.1f}%"})
    return pd.DataFrame(rows)

def find_duplicates(open_df: pd.DataFrame) -> pd.DataFrame:
    col_ep   = find_col(open_df, ["ENTRY PRICE", "Entry Price"])
    col_date = find_col(open_df, ["ENTRY DATE", "Entry Date"])
    if not col_ep:
        return pd.DataFrame()
    rows = []
    for scrip, g in open_df.groupby("SCRIP"):
        g2    = g.sort_values(col_ep)
        ep    = g2[col_ep].tolist()
        dates = g2[col_date].tolist() if col_date else [pd.NaT] * len(ep)
        seen  = {}
        for p, d in zip(ep, dates):
            pr    = round(p, 2)
            match = next((k for k in seen if abs(k - pr) / k < DUP_TOL), None)
            if match:
                rows.append({"SCRIP": scrip,
                             "Price 1 (Rs)": match, "Date 1": seen[match],
                             "Price 2 (Rs)": pr,    "Date 2": d,
                             "Diff %": f"{abs(match-pr)/match*100:.2f}%",
                             "Type": "Exact" if match == pr else "Near-Dup (<1%)"})
            else:
                seen[pr] = d
    return pd.DataFrame(rows)

def sl_check(port: pd.DataFrame) -> pd.DataFrame:
    """
    BUG FIX: previously sl_check returned empty df when SL/CMP were NaN
    because column lookup in build_portfolio failed silently.
    Now build_portfolio uses find_col() so SL/HH/Pct_HH are always populated.
    """
    df = port.copy()
    # Only keep rows where we actually have SL and CMP values
    mask = df["SL"].notna() & df["CMP"].notna() & (df["SL"] > 0)
    df   = df[mask].copy()
    if df.empty:
        return df

    df["SL_Hit"]     = df["CMP"] < df["SL"]
    df["Pct_SL"]     = ((df["CMP"] - df["SL"]) / df["SL"] * 100).round(2)
    # Pct_HH is stored as decimal fraction: -0.31 means -31%
    df["Pct_HH_pct"] = (df["Pct_HH"] * 100).round(2)
    return df.sort_values("Pct_SL")

def top_bottom_10(port: pd.DataFrame):
    n   = max(1, round(len(port) * 0.10))
    srt = port.sort_values("Ret_Pct")
    return srt.head(n).copy(), srt.tail(n).iloc[::-1].copy()

def compute_mtm(port: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prog = st.progress(0, text="Fetching previous closes…")
    for i, (_, r) in enumerate(port.iterrows()):
        pc = prev_close(r["SCRIP"]) or r["CMP"]
        dc = r["CMP"] - pc
        rows.append({"SCRIP": r["SCRIP"], "Qty": r["Qty"],
                     "Prev Close": round(pc, 2), "CMP": round(r["CMP"], 2),
                     "Day Chg (Rs)": round(dc, 2),
                     "Day Chg (%)": round(dc / pc * 100 if pc else 0, 2),
                     "Day PnL (Rs)": round(dc * r["Qty"], 2),
                     "Status": "Gain" if dc > 0 else ("Loss" if dc < 0 else "Flat")})
        prog.progress((i + 1) / len(port))
    prog.empty()
    return pd.DataFrame(rows).sort_values("Day PnL (Rs)", ascending=False)

# ─────────────────────── STYLING HELPERS ─────────────────────────
def mkbar(df, x, y, title, scale="RdYlGn"):
    fig = px.bar(df, x=x, y=y, color=y, color_continuous_scale=scale,
                 text_auto=".0f", title=title)
    fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_tickangle=-40)
    return fig

def mkpie(df, values, names, title):
    fig = px.pie(df, values=values, names=names, title=title, hole=0.35)
    fig.update_layout(**PLOT_LAYOUT, height=380)
    return fig

def fmt(df, rename, fmts):
    return df.rename(columns=rename).style.format(fmts)

def color_pnl_cell(val):
    """Green bg + WHITE text for positive, red bg + white text for negative."""
    try:
        v = float(val)
        if v > 0:   return "background-color:#14532d;color:#ffffff;font-weight:600"
        if v < 0:   return "background-color:#450a0a;color:#ffffff;font-weight:600"
    except Exception:
        pass
    return ""

def apply_pnl_color(styler, cols):
    """Apply green/red coloring to specified columns."""
    return styler.map(color_pnl_cell, subset=cols) if hasattr(styler, "map") else styler.applymap(color_pnl_cell, subset=cols)

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
uploaded = None
use_live = True
fetch_mtm = True

with st.sidebar:
    st.title("📊 Pyramiding Dashboard")
    st.markdown("---")
    auto_refresh = st.toggle("⏱️ Auto Refresh (1 min)", value=False)
    
    if auto_refresh:
        if st_autorefresh:
            st_autorefresh(interval=60 * 1000, key="dataframerefresh")
        else:
            st.sidebar.error("pip install streamlit-autorefresh missing!")

    if st.button("🔄 Clear Cache", type="secondary"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Buy on breakout → +5% each add-on\nSL = 40% below highest high")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
st.title("📈 Pyramiding Portfolio Dashboard")

import os
if not uploaded:
    default_file = "pyramidingnew.xlsx"
    if os.path.exists(default_file):
        uploaded = default_file
        st.info(f"📁 Auto-loaded default file: **{default_file}**")
    else:
        st.info("👈 Upload your portfolio Excel/CSV from the sidebar to begin.")
        st.stop()

# ── Load & Parse ─────────────────────────────────────────────────
raw              = load_raw(uploaded)
open_df, exited_df = parse_sheet(raw)

if open_df.empty:
    st.error("No open BUY positions found. Check your file has a 'Buy' column with value 'BUY'.")
    st.stop()

# ── Build portfolio (cached in session) ──────────────────────────
if isinstance(uploaded, str):
    import os
    st_stat = os.stat(uploaded)
    cache_key = f"{uploaded}_{st_stat.st_size}_{st_stat.st_mtime}_{use_live}"
else:
    cache_key = f"{uploaded.name}_{uploaded.size}_{use_live}"
if st.session_state.get("port_key") != cache_key:
    with st.spinner("Building portfolio…"):
        port = build_portfolio(open_df, use_live)
    st.session_state.port     = port
    st.session_state.port_key = cache_key
    st.session_state.pop("mtm_df", None)
else:
    port = st.session_state.port

real_total, real_by_scrip = realized_pnl(exited_df)

# ── Key numbers ──────────────────────────────────────────────────
total_inv  = port["Inv"].sum()
total_val  = port["Cur_Val"].sum()
unrealized = port["Unrealized"].sum()
net_pnl    = unrealized + real_total
net_ret    = net_pnl / total_inv * 100 if total_inv else 0
profit_n   = int((port["Unrealized"] > 0).sum())
loss_n     = int((port["Unrealized"] < 0).sum())
best       = port.loc[port["Unrealized"].idxmax()]
worst      = port.loc[port["Unrealized"].idxmin()]
top_pyr    = port.loc[port["N_Entries"].idxmax()]

# ════════════════════════════════════════════════════════════════
# SUMMARY BANNER
# ════════════════════════════════════════════════════════════════
st.markdown("## Portfolio Summary")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("💰 Net P&L",        f"Rs {net_pnl:,.0f}",    f"{net_ret:+.2f}%")
c2.metric("📦 Portfolio Value", f"Rs {total_val:,.0f}",  f"Invested Rs {total_inv:,.0f}")
c3.metric("📊 Unrealized P&L", f"Rs {unrealized:,.0f}", f"{unrealized/total_inv*100:+.2f}%" if total_inv else "—")
c4.metric("🏦 Realized P&L",   f"Rs {real_total:,.0f}", f"{exited_df['SCRIP'].nunique() if not exited_df.empty else 0} stocks exited")
c5.metric("🔢 Open Stocks",    f"{len(port)}",           f"Pyramiding: {(port['N_Entries']>1).sum()}")

c6,c7,c8,c9,c10 = st.columns(5)
c6.metric("🟢 Profitable",       f"{profit_n}")
c7.metric("🔴 Loss",              f"{loss_n}")
c8.metric("🏆 Best",  best["SCRIP"],  f"Rs {best['Unrealized']:,.0f} ({best['Ret_Pct']:+.1f}%)")
c9.metric("⚠️ Worst", worst["SCRIP"], f"Rs {worst['Unrealized']:,.0f} ({worst['Ret_Pct']:+.1f}%)")
c10.metric("🔝 Most Pyramided",   top_pyr["SCRIP"], f"{top_pyr['N_Entries']} entries")
st.markdown("---")

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
(t_nb, t_sl, t_dup, t_miss) = st.tabs([
    "🎯 Next Buy","🛑 Near SL","🔁 Duplicates","❓ Missing Levels",
])
# ─────────────────── NEXT BUY ────────────────────────────────────
with t_nb:
    st.subheader("🎯 Next Pyramid Buy Levels (+5% from last entry)")
    nb = port[["SCRIP","Mcap","N_Entries","Last_Entry","Next_Buy",
               "CMP","Gap_Next","Unrealized","Ret_Pct"]].copy()
    nb["Signal"] = nb["Gap_Next"].apply(
        lambda g: "🟢 BUY NOW"   if g <= 0
             else "⚡ < 2% away" if g < 2
             else "📌 2–5% away" if g < 5
             else "⏳ Waiting")
    nb = nb.sort_values("Gap_Next")

    flt = st.radio("Show:", ["All","Ready + Close (<=5%)","< 2% only"], horizontal=True)
    if flt == "Ready + Close (<=5%)": nb = nb[nb["Gap_Next"] <= 5]
    elif flt == "< 2% only":          nb = nb[nb["Gap_Next"] <= 2]

    st.caption(f"{len(nb)} stocks shown")
    st.dataframe(
        nb.rename(columns={"N_Entries":"Entries","Last_Entry":"Last Buy (Rs)",
                            "Next_Buy":"Next Buy (Rs)","Gap_Next":"Gap (%)",
                            "Unrealized":"PnL (Rs)","Ret_Pct":"Return %"})
        .style.format({"Last Buy (Rs)":"Rs {:,.2f}","Next Buy (Rs)":"Rs {:,.2f}",
                       "CMP":"Rs {:,.2f}","Gap (%)":"{:.2f}%",
                       "PnL (Rs)":"Rs {:,.0f}","Return %":"{:.2f}%"}),
        use_container_width=True, height=550, hide_index=True)

# ─────────────────── NEAR SL ─────────────────────────────────────
with t_sl:
    st.subheader("🛑 Stop Loss Monitor — 40% from Highest High")

    sl_df = sl_check(port)

    if sl_df.empty:
        st.warning("SL / CMP data not available. Enable **'Fetch live prices (Yahoo Finance)'** in the sidebar to auto-calculate SL/CMP, OR check that your file has 'SL', 'CMP', and 'HighestHigh' columns.")
    else:
        breached = sl_df[sl_df["SL_Hit"]]
        near     = sl_df[~sl_df["SL_Hit"] & (sl_df["Pct_HH_pct"] < -34)]
        safe     = sl_df[~sl_df["SL_Hit"] & (sl_df["Pct_HH_pct"] >= -34)]

        m1,m2,m3 = st.columns(3)
        m1.metric("⛔ SL Breached",          len(breached), "EXIT NOW" if len(breached) else "")
        m2.metric("⚠️ Near SL (>34% off HH)", len(near),    "Review"   if len(near) else "")
        m3.metric("✅ Safe",                  len(safe))

        SL_COLS = [c for c in ["SCRIP","Mcap","CMP","SL","HH","Pct_HH_pct",
                                "Unrealized","Ret_Pct"] if c in sl_df.columns]
        SL_REN  = {"HH":"Highest High","Pct_HH_pct":"% from HH",
                   "Unrealized":"PnL (Rs)","Ret_Pct":"Return %"}
        SL_FMT  = {"CMP":"Rs {:,.2f}","SL":"Rs {:,.2f}","Highest High":"Rs {:,.2f}",
                   "% from HH":"{:.1f}%","PnL (Rs)":"Rs {:,.0f}","Return %":"{:.1f}%"}

        if not breached.empty:
            st.error("### ⛔ SL BREACHED — EXIT IMMEDIATELY")
            styled_breach = breached[SL_COLS].rename(columns=SL_REN).style.format(SL_FMT)
            styled_breach = styled_breach.map(lambda _: "background-color:#7f1d1d;color:#ffffff;font-weight:700") if hasattr(styled_breach, "map") else styled_breach.applymap(lambda _: "background-color:#7f1d1d;color:#ffffff;font-weight:700")
            st.dataframe(styled_breach, use_container_width=True, hide_index=True)
        else:
            st.success("No SL breached right now.")

        st.markdown("---")
        st.warning("### ⚠️ Approaching Stop Loss (>34% below highest high)")
        if not near.empty:
            styled_near = near[SL_COLS].rename(columns=SL_REN).style.format(SL_FMT)
            styled_near = styled_near.map(lambda _: "background-color:#431407;color:#ffffff;font-weight:600") if hasattr(styled_near, "map") else styled_near.applymap(lambda _: "background-color:#431407;color:#ffffff;font-weight:600")
            st.dataframe(styled_near, use_container_width=True, hide_index=True)
        else:
            st.success("No stocks approaching SL threshold.")

        with st.expander("✅ Safe Stocks"):
            st.dataframe(safe[SL_COLS].rename(columns=SL_REN).style.format(SL_FMT),
                         use_container_width=True, hide_index=True)

        # Bar chart
        viz = sl_df[["SCRIP","Pct_HH_pct"]].sort_values("Pct_HH_pct").copy()
        viz["Zone"] = viz["Pct_HH_pct"].apply(
            lambda v: "SL Breached" if v < -40 else ("Near SL" if v < -34 else "Safe"))
        fig_sl = px.bar(viz, x="SCRIP", y="Pct_HH_pct", color="Zone",
                        color_discrete_map={"SL Breached":"#ef4444","Near SL":"#f97316","Safe":"#22c55e"},
                        title="Distance from Highest High (%) — All Stocks")
        fig_sl.add_hline(y=-40, line_dash="dash", line_color="#ef4444",
                         annotation_text="SL boundary (−40%)")
        fig_sl.update_layout(**PLOT_LAYOUT, height=420, xaxis_tickangle=-45)
        st.plotly_chart(fig_sl, use_container_width=True)

# ─────────────────── DUPLICATES ──────────────────────────────────
with t_dup:
    st.subheader("🔁 Duplicate / Near-Duplicate Entries")
    dup_df = find_duplicates(open_df)
    if not dup_df.empty:
        st.error(f"**{len(dup_df)} duplicate entries** — capital deployed at same pyramid level twice.")
        st.dataframe(
            dup_df.style.format({"Price 1 (Rs)":"Rs {:,.2f}","Price 2 (Rs)":"Rs {:,.2f}"}),
            use_container_width=True, hide_index=True)
        st.markdown("> **Fix:** Before every add-on, check all existing entries. Within 1% of any entry → skip it.")
    else:
        st.success("✅ No duplicate entries found!")

# ─────────────────── MISSING LEVELS ──────────────────────────────
with t_miss:
    st.subheader("❓ Missing Pyramid Levels")
    st.info("A missing level = a +5% step skipped between your first and last entry.")
    miss_df = find_missing_levels(port)
    if not miss_df.empty:
        st.warning(f"**{len(miss_df)} missing levels** across **{miss_df['SCRIP'].nunique()} stocks**")
        st.dataframe(
            miss_df.style.format({"Expected (Rs)":"Rs {:,.2f}","Prev Level (Rs)":"Rs {:,.2f}"}),
            use_container_width=True, hide_index=True)
        cnt = miss_df.groupby("SCRIP").size().reset_index(name="Missing Levels")
        fig_m = px.bar(cnt.sort_values("Missing Levels",ascending=False),
                       x="SCRIP", y="Missing Levels", color="Missing Levels",
                       color_continuous_scale="Oranges", title="Missing Pyramid Levels per Stock")
        fig_m.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_m, use_container_width=True)
        st.markdown("> **Fix:** Use GTT orders at each +5% level right after every entry.")
    else:
        st.success("✅ No missing levels — perfect pyramid execution!")

st.markdown("---")
st.caption("Pyramiding Dashboard · Prices via Yahoo Finance · Data processed locally")
