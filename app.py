"""
US ETF Momentum - Strategy Lab (Streamlit dashboard)
Set parameters in the sidebar, run the backtest, and explore the results.

Start it with run_dashboard.bat (or: streamlit run app.py)
Analytical tool only - not investment advice.
"""
import os
import sys
from datetime import datetime
from dataclasses import replace, asdict
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import us_etf_momentum as eng  # noqa: E402

st.set_page_config(page_title="US ETF Momentum Lab", page_icon="📈", layout="wide")

BLUE, GREY, GOLD, RED, GREEN = "#1f5fbf", "#9aa0a6", "#c9a227", "#c0392b", "#2e8b57"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_prices(market: str, refresh_token: int):
    eng.set_profile(market)
    csv = os.environ.get(f"ETF_PRICES_CSV_{market.upper()}") or os.environ.get("ETF_PRICES_CSV")
    return eng.load_prices(refresh=refresh_token > 0, csv_path=csv)


def run(cfg, close):
    eq, trades = eng.run_backtest(close, cfg)
    stats = eng.perf_stats(eq["Equity"])
    bm = eng.benchmarks(close, eq.index, cfg.capital)
    bm_stats = {k: eng.perf_stats(v) for k, v in bm.items()}
    return dict(cfg=cfg, eq=eq, trades=trades, stats=stats, bm=bm, bm_stats=bm_stats)


def cat_key_for(m):
    return f"cat_table_{m}"


def pct(x, d=2):
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------
# Sidebar - parameters
# --------------------------------------------------------------------------
if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if "runs_by_market" not in st.session_state:
    st.session_state.runs_by_market = {}

st.sidebar.title("⚙️ Strategy parameters")
market = st.sidebar.selectbox(
    "Market", list(eng.PROFILES), format_func=lambda m: eng.PROFILES[m]["label"],
    help="Ireland = Ireland-domiciled UCITS ETFs/ETCs on the London Stock Exchange, "
         "converted to USD. Often used by non-US investors (lower US dividend withholding, "
         "no US estate-tax exposure). India = NSE-listed ETFs in INR, one category per sector.")
eng.set_profile(market)
st.session_state.runs = st.session_state.runs_by_market.setdefault(market, {})

def do_quick_update():
    """Pull the last few days (incl. today's live bar) for this market and the selected global symbols."""
    errs = []
    try:
        eng.quick_update_prices()
    except Exception as e:
        errs.append(f"ETF prices: {e}")
    gsel = st.session_state.get("gl_sel")
    if gsel:
        try:
            eng.quick_update_global(gsel)
        except Exception as e:
            errs.append(f"Global markets: {e}")
    get_prices.clear()
    get_global.clear()
    st.session_state["last_update"] = datetime.now()
    return errs


@st.cache_data(show_spinner=False)
def get_global(symbols: tuple, token: int):
    return eng.load_global(list(symbols), refresh=token > 0)


with st.sidebar.expander("Data", expanded=True):
    if st.button("⚡ Update to latest prices", width='stretch', type="primary",
                 help="Fetches only the last 5 days, including today's live bar (Yahoo, ~15 min delayed). "
                      "Takes a few seconds."):
        with st.spinner("Updating latest prices..."):
            errs = do_quick_update()
        for e_ in errs:
            st.error(e_)
    auto_upd = st.toggle("Auto-update while the app is open", key="auto_upd")
    upd_every = st.selectbox("Every", [5, 15, 30, 60], index=1, format_func=lambda m: f"{m} minutes",
                             disabled=not auto_upd, key="upd_every")
    lu = st.session_state.get("last_update")
    st.caption(f"Last update: **{lu:%d %b %H:%M}** (your PC time)" if lu else "Last update: not yet this session")
    if st.button("Full re-download (all history)", width='stretch',
                 help="Rebuilds the whole price history - use occasionally, e.g. after dividends/splits."):
        st.session_state.refresh += 1
        get_prices.clear()

if auto_upd:
    @st.fragment(run_every=60)
    def _auto_updater():
        lu_ = st.session_state.get("last_update")
        if lu_ is None or (datetime.now() - lu_).total_seconds() >= upd_every * 60:
            do_quick_update()
            st.rerun(scope="app")
    _auto_updater()

try:
    with st.spinner("Loading price data (first time downloads ~20 years)..."):
        close = get_prices(market, st.session_state.refresh)
except SystemExit as e:
    st.error(f"Could not load prices: {e}")
    st.stop()

last_day = close.index[-1].date()
st.sidebar.caption(f"Prices up to **{last_day}** · {close.shape[1]} tickers")

sb = st.sidebar
run_name = sb.text_input("Run name", value=f"Run {len(st.session_state.runs) + 1}")

sb.subheader("Backtest window")
c1, c2 = sb.columns(2)
start = c1.date_input("Start", value=pd.Timestamp(eng.PROFILES[market]["default_start"]).date(),
                      min_value=date(2006, 1, 1), max_value=last_day, key=f"start_{market}")
end = c2.date_input("End", value=last_day, min_value=date(2007, 1, 1), max_value=last_day)
capital = sb.number_input(f"Capital ({eng.CCY})", min_value=1000,
                          value=1_000_000 if market == "india" else 100_000, step=10_000, key=f"cap_{market}")

sb.subheader("Portfolio")
c1, c2 = sb.columns(2)
n_hold = c1.number_input("No. of ETFs", 1, 20, 5)
exit_rank = c2.number_input("Exit rank", 1, 40, 10)
rank_label = sb.selectbox("Rank criteria", ["Sharpe Return", "Return Percent"])

sb.subheader("Lookback periods (days / weight)")
periods, weights = [], []
defaults = [(252, 2, True), (90, 1, True), (60, 1, True), (120, 1, False)]
for i, (p, w, on) in enumerate(defaults):
    c0, c1, c2 = sb.columns([0.8, 1.2, 1])
    use = c0.checkbox(" ", value=on, key=f"use{i}", label_visibility="collapsed")
    pv = c1.number_input("Days", 5, 504, p, key=f"p{i}", label_visibility="collapsed")
    wv = c2.number_input("Wt", 0.0, 10.0, float(w), 0.5, key=f"w{i}", label_visibility="collapsed")
    if use and wv > 0:
        periods.append(int(pv))
        weights.append(float(wv))
if not periods:
    sb.error("Select at least one period.")

sb.subheader("Rebalance")
c1, c2 = sb.columns(2)
rebalance = c1.selectbox("Frequency", ["weekly", "biweekly", "monthly"])
weekday = c2.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri"], index=2,
                       disabled=rebalance == "monthly")

sb.subheader("Entry filters")
entry_cond = sb.selectbox("Entry condition", list(eng.ENTRY_CONDITIONS),
                          index=list(eng.ENTRY_CONDITIONS).index("Close above 200 EMA"),
                          help="Technical condition an ETF must meet (on the signal day) before it can be bought.")
use_retr = sb.checkbox("Within X% of 52-week high", value=True)
retr = sb.slider("Retracement limit (%)", 5, 50, 20, disabled=not use_retr)
c1, c2 = sb.columns([1, 2])
use_rs = c1.checkbox(f"RS vs {eng.BENCHMARK}", value=True,
                     help=f"Relative strength: ETF price divided by {eng.BENCHMARK} (S&P 500).")
rs_rule = c2.selectbox("RS rule", list(eng.RS_RULES), disabled=not use_rs, label_visibility="collapsed")

sb.subheader("Exits & costs")
exit_cond = sb.selectbox("Exit condition", list(eng.EXIT_CONDITIONS),
                         index=list(eng.EXIT_CONDITIONS).index("Close below 200 EMA"),
                         help="Checked at each rebalance; a held ETF meeting it is sold. Rank exit and stop loss still apply.")
use_stop = sb.checkbox("Stop loss", value=True)
stop = sb.slider("Stop loss (%)", 3, 30, 12, disabled=not use_stop)
use_cash = sb.checkbox(f"Park empty slots in {eng.CASH_ETF}", value=True)
cost = sb.number_input("Cost per side (bps)", 0.0, 100.0, 5.0, 1.0)

sb.subheader("ETF selection")
all_t = [t for t in eng.UNIVERSE if t in close.columns]
label_of = lambda t: f"{t} · {eng.NAMES[t]}" if eng.NAMES.get(t) else t
mode = sb.radio("Universe", ["All ETFs", "Select ETFs manually"], horizontal=True, key=f"mode_{market}")
included, excluded = [], []
if mode == "Select ETFs manually":
    cats_all = sorted({eng.CATEGORY.get(t, t) for t in all_t})
    pick_cats = sb.multiselect("Quick add: whole categories", cats_all, key=f"pickcat_{market}",
                               help="Every ETF in the chosen categories is added to the selection below.")
    sel_key = f"sel_{market}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = []
    if pick_cats and sb.button("Add these categories", width='stretch'):
        st.session_state[sel_key] = sorted(set(st.session_state[sel_key]) |
                                           {t for t in all_t if eng.CATEGORY.get(t, t) in pick_cats})
        st.rerun()
    included = sb.multiselect("ETFs to test", all_t, key=sel_key, format_func=label_of,
                              placeholder="Choose ETFs...")
    c1, c2 = sb.columns(2)
    if c1.button("Select all", width='stretch'):
        st.session_state[sel_key] = list(all_t)
        st.rerun()
    if c2.button("Clear", width='stretch'):
        st.session_state[sel_key] = []
        st.rerun()
    if included:
        sb.caption(f"{len(included)} ETF(s) selected"
                   + (f" · fewer than {int(n_hold)} holdings - spare slots go to {eng.CASH_ETF}"
                      if len(included) < n_hold else ""))
    else:
        sb.warning("Select at least one ETF.")
else:
    excluded = sb.multiselect("Exclude ETFs", all_t, default=[], key=f"excl_{market}", format_func=label_of)

with sb.expander("➕ Add your own ticker"):
    st.caption("Any Yahoo Finance symbol, e.g. VWRA.L, XLK, ARKK, or NSE ETFs like BANKBEES.NS. "
               + ("Prices are converted to USD." if eng.TO_USD else ""))
    new_t = st.text_input("Ticker(s), comma-separated", key=f"newt_{market}")
    cc1, cc2 = st.columns(2)
    new_asset = cc1.text_input("Asset class", value="Custom", key=f"newa_{market}")
    new_cat = cc2.text_input("Category", value="", key=f"newc_{market}",
                             placeholder="blank = its own category")
    new_name = st.text_input("Name (optional)", key=f"newn_{market}")
    if st.button("Download & add", width='stretch', disabled=not new_t.strip()):
        ents = {t.strip().upper(): (new_asset or "Custom", new_cat.strip() or t.strip().upper(), new_name)
                for t in new_t.split(",") if t.strip()}
        with st.spinner("Downloading..."):
            try:
                added, failed = eng.add_custom_tickers(ents)
            except Exception as e:
                added, failed = [], list(ents)
                st.error(f"Download error: {e}")
        if added:
            st.success(f"Added: {', '.join(added)}")
            get_prices.clear()
            st.session_state.pop(cat_key_for(market), None)
        if failed:
            st.error(f"No data found for: {', '.join(failed)} (check the Yahoo symbol)")
        if added:
            st.rerun()
    mine = eng.custom_tickers()
    if mine:
        st.write("Your added tickers:")
        for t, (a_, c_, n_) in mine.items():
            r1, r2 = st.columns([3, 1])
            r1.write(f"**{t}** · {c_}" + (f" · {n_}" if n_ else ""))
            if r2.button("Remove", key=f"rm_{market}_{t}"):
                eng.remove_custom_ticker(t)
                st.session_state.pop(cat_key_for(market), None)
                sel = st.session_state.get(f"sel_{market}", [])
                st.session_state[f"sel_{market}"] = [x for x in sel if x != t]
                st.rerun()

sb.subheader("ETF Group")
use_group = sb.checkbox("Only the best ETF per category", value=True,
                        help="ETFs that move together share a category (e.g. gold & silver = Precious Metals). "
                             "With this on, the strategy holds at most N ETFs from any one category.")
c1, c2 = sb.columns(2)
max_cat = c1.number_input("Max per category", 1, 5, 1, disabled=not use_group)
cat_buf = c2.number_input("Swap buffer", 0, 5, 1, disabled=not use_group,
                          help="A held ETF is swapped for a better one in its category only when it drops "
                               "below rank (max + buffer) within that category. Higher = fewer swaps.")
cat_key = cat_key_for(market)
if cat_key not in st.session_state:
    st.session_state[cat_key] = pd.DataFrame(
        {"ETF": all_t, "Name": [eng.NAMES.get(t, "") for t in all_t],
         "Asset class": [eng.UNIVERSE[t] for t in all_t],
         "Category": [eng.CATEGORY.get(t, t) for t in all_t]})
with sb.expander("Edit categories"):
    st.caption("Type a category name to regroup ETFs. ETFs with the same name count as one category.")
    edited = st.data_editor(st.session_state[cat_key], hide_index=True, height=300, key=f"cat_editor_{market}",
                            disabled=["ETF", "Name", "Asset class"], width='stretch')
    if st.button("Reset to default categories", width='stretch'):
        del st.session_state[cat_key]
        st.rerun()
cat_over = tuple((r.ETF, str(r.Category).strip() or r.ETF) for r in edited.itertuples()
                 if (str(r.Category).strip() or r.ETF) != eng.CATEGORY.get(r.ETF, r.ETF))

cfg = eng.Config(
    start=str(start), end=str(end), capital=float(capital), n_holdings=int(n_hold),
    exit_rank=int(exit_rank), rank_criteria="sharpe" if rank_label.startswith("Sharpe") else "return",
    periods=tuple(periods), weights=tuple(weights), retracement=retr / 100 if use_retr else None,
    use_rs=use_rs, rs_rule=rs_rule, entry_condition=entry_cond, exit_condition=exit_cond, stop_loss=stop / 100 if use_stop else None,
    rebalance=rebalance, weekday=["Mon", "Tue", "Wed", "Thu", "Fri"].index(weekday),
    cost_bps=float(cost), use_cash_etf=use_cash, exclude=tuple(excluded), include=tuple(included),
    max_per_category=int(max_cat) if use_group else 0, category_buffer=int(cat_buf),
    categories=cat_over, name=run_name)

run_clicked = sb.button("▶ Run backtest", type="primary", width='stretch',
                        disabled=not periods or (mode == "Select ETFs manually" and not included))
if run_clicked:
    with st.spinner("Running backtest..."):
        st.session_state.runs[run_name] = run(cfg, close)
        st.session_state.current = run_name

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("📈 ETF Momentum Lab")
st.caption(f"Market: **{eng.PROFILES[market]['label']}** · benchmark {eng.BENCHMARK} · cash {eng.CASH_ETF}"
           + (" · prices converted to USD" if eng.TO_USD else ""))

view = st.radio("View", ["📊 Backtest", "📈 Sector comparison", "🌍 Global markets", "🇮🇳 India ETF explorer"],
                horizontal=True,
                label_visibility="collapsed", key="view")


def short(t):
    return t.split(".")[0]


def first_valid_norm(df):
    """Rebase each column to 0% at its first valid price in the window."""
    base = df.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan)
    return (df / base - 1) * 100


def spread_labels(vals, gap):
    """Nudge end-of-line label positions apart so they don't overlap."""
    order = sorted(vals, key=lambda kv: kv[1])
    pos, last = {}, None
    for k, v in order:
        y = v if last is None or v - last >= gap else last + gap
        pos[k], last = y, y
    return pos


def end_label_chart(df, base, height, ysuffix, ytitle, fmt, gap_frac, lab, yrange=None, bands=None):
    """Multi-line chart with TradingView-style coloured value tags at the right edge."""
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    fig = go.Figure()
    if bands:
        for y0, y1, colr in bands:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=colr, line_width=0)
    finals, colors = {}, {}
    for i, col in enumerate(df.columns):
        ser = df[col].dropna()
        if ser.empty:
            continue
        is_b = col == base
        colr = "#888888" if is_b else palette[i % len(palette)]
        colors[col] = colr
        fig.add_trace(go.Scatter(x=ser.index, y=ser, mode="lines", name=lab(col),
                                 line=dict(color=colr, width=3.2 if is_b else 1.8, dash="dash" if is_b else "solid"),
                                 hovertemplate=f"<b>{lab(col)}</b> %{{x|%d %b %Y}}: %{{y:.2f}}{ysuffix}<extra></extra>"))
        finals[col] = ser.iloc[-1]
    span = (df.max().max() - df.min().min()) if yrange is None else (yrange[1] - yrange[0])
    pos = spread_labels(list(finals.items()), max(span, 1.0) * gap_frac)
    for col, v in finals.items():
        fig.add_annotation(x=df.index[-1], y=pos[col], text=f"<b>{lab(col)}</b> {fmt(v)}", xanchor="left",
                           xshift=8, showarrow=False, font=dict(color="white", size=11), bgcolor=colors[col], borderpad=2)
    fig.update_layout(height=height, showlegend=False, hovermode="closest", yaxis_ticksuffix=ysuffix,
                      yaxis_title=ytitle, margin=dict(t=20, r=200, b=10),
                      xaxis=dict(range=[df.index[0], df.index[-1]]))
    if yrange:
        fig.update_yaxes(range=yrange)
    return fig, finals, colors


@st.cache_data(show_spinner=False)
def get_nse_list(token: int):
    return eng.fetch_nse_etf_list(refresh=token > 0)


@st.cache_data(show_spinner=False)
def get_india_all(symbols: tuple, token: int, quick_token: int):
    return eng.load_india_all(list(symbols), refresh=False, quick=quick_token > 0)


if view == "🇮🇳 India ETF explorer":
    st.subheader("All NSE ETFs (plus REITs / InvITs)")
    st.caption("Every ETF listed on NSE, grouped into categories, ranked overall and within its category. "
               "Independent of the Market chosen in the left panel; the momentum score uses your left-panel "
               "lookback periods, weights and rank criteria.")
    b1, b2, b3 = st.columns(3)
    if b1.button("🔄 Refresh ETF list from NSE", width='stretch'):
        st.session_state["nse_tok"] = st.session_state.get("nse_tok", 0) + 1
        get_nse_list.clear()
    if b2.button("⚡ Update prices (last 5 days)", width='stretch'):
        st.session_state["ia_quick"] = st.session_state.get("ia_quick", 0) + 1
        get_india_all.clear()
    try:
        lst = get_nse_list(st.session_state.get("nse_tok", 0))
    except Exception as e:
        lst = None
        st.error(f"Couldn't download the ETF list from NSE ({e}).")
    with st.expander("Use your own ETF list file instead", expanded=lst is None):
        st.caption("On nseindia.com go to Market Data → Exchange Traded Funds and click 'Download (.csv)', "
                   "then upload it here. Any CSV with a Symbol column works.")
        up = st.file_uploader("ETF list CSV", type=["csv"], key="nse_upload")
        if up is not None:
            try:
                lst = eng.save_etf_list_upload(pd.read_csv(up))
                get_nse_list.clear()
                st.success(f"Loaded {len(lst)} symbols from your file.")
            except Exception as e:
                st.error(f"Could not read that file: {e}")
    if lst is None or lst.empty:
        st.stop()
    with st.expander("Edit categories (optional)"):
        st.caption("Categories are guessed from each ETF's name and underlying index - fix any here.")
        cat_opts = sorted(set(lst.Category) | {c_ for _, c_ in eng.INDIA_CATEGORY_RULES} | {"REIT / InvIT", "Other Equity"})
        lst = st.data_editor(lst, hide_index=True, height=300, key="ia_cat_editor", width='stretch',
                             disabled=["Symbol", "Name", "Underlying"],
                             column_config={"Category": st.column_config.SelectboxColumn(options=cat_opts)})
    syms = tuple(sorted(lst.Symbol))
    with st.spinner(f"Loading prices for {len(syms)} symbols (first time takes a few minutes)..."):
        try:
            ia_close, ia_vol = get_india_all(syms, 0, st.session_state.get("ia_quick", 0))
        except Exception as e:
            st.error(f"Price download failed: {e}")
            st.stop()
    tbl = eng.india_etf_table(lst, ia_close, ia_vol, cfg)

    f1, f2, f3, f4 = st.columns([1.4, 2.2, 1.2, 1.2])
    rank_by = f1.selectbox("Rank by", ["Momentum score", "1W", "1M", "3M", "6M", "1Y", "3Y"], key="ia_rankby",
                           help="Momentum score = the same score the backtest uses (left-panel settings).")
    cats_f = f2.multiselect("Categories", sorted(tbl.Category.unique()), key="ia_cats", placeholder="All categories")
    min_liq = f3.number_input("Min avg value (₹ Cr/day)", 0.0, 1000.0, 0.0, 0.5, key="ia_liq",
                              help="Hide thinly traded ETFs. 1-2 Cr/day is a sensible floor for real trading.")
    search = f4.text_input("Search", key="ia_search", placeholder="e.g. gold, bank, nifty 50")
    key = "Score" if rank_by == "Momentum score" else rank_by
    t = tbl.copy()
    if min_liq > 0:
        t = t[t["Avg value 1M (₹ Cr)"].fillna(0) >= min_liq]
    t["Overall rank"] = t[key].rank(ascending=False, method="min")
    t["Category rank"] = t.groupby("Category")[key].rank(ascending=False, method="min")
    if cats_f:
        t = t[t.Category.isin(cats_f)]
    if search:
        q = search.lower()
        t = t[t.apply(lambda r: q in f"{r.Symbol} {r['ETF name']} {r.Underlying} {r.Category}".lower(), axis=1)]
    t = t.sort_values(["Overall rank", "Symbol"], na_position="last")
    no_data = int(tbl["1W"].isna().sum()) if "1W" in tbl else len(tbl)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Symbols in list", len(tbl))
    m2.metric("With price data", len(tbl) - no_data)
    m3.metric("Categories", tbl.Category.nunique())
    m4.metric("Showing", len(t))
    show_cols = ["Overall rank", "Category rank", "Symbol", "ETF name", "Category", "1W", "1M", "6M", "1Y", "3Y",
                 "Score", "Avg value 1M (₹ Cr)", "Price", "Last date"]
    show_cols = [c_ for c_ in show_cols if c_ in t.columns]
    rets_c = [c_ for c_ in ["1W", "1M", "6M", "1Y", "3Y"] if c_ in t.columns]
    st.dataframe(t[show_cols].style.format({**{c_: "{:+.2f}%" for c_ in rets_c}, "Score": "{:.2f}",
                                            "Avg value 1M (₹ Cr)": "{:,.2f}", "Price": "{:,.2f}",
                                            "Overall rank": "{:.0f}", "Category rank": "{:.0f}"}, na_rep="–")
                 .background_gradient(subset=rets_c, cmap="RdYlGn", vmin=-15, vmax=15),
                 width='stretch', height=640, hide_index=True)
    st.caption(f"Overall rank = position among all shown ETFs by **{rank_by}**; Category rank = position within its "
               "category. Returns use Yahoo adjusted prices. Liquid ETFs that pay daily dividends as extra units "
               "(e.g. LIQUIDBEES) show near-zero price returns. '–' = not enough history or no data on Yahoo "
               f"({no_data} symbols).")
    st.download_button("Download this table (CSV)", t[show_cols].to_csv(index=False), "india_etfs.csv", "text/csv")

    st.subheader("Category leaders")
    lead = t.dropna(subset=[key]).sort_values(key, ascending=False).groupby("Category").head(1)
    lead = lead.sort_values(key, ascending=False)
    if not lead.empty:
        figl = px.bar(lead, x=key, y="Category", orientation="h", text="Symbol",
                      color=lead[key] > 0, color_discrete_map={True: GREEN, False: RED},
                      title=f"Best ETF in each category by {rank_by}")
        figl.update_layout(showlegend=False, height=max(350, 26 * len(lead)), yaxis_title=None,
                           xaxis_ticksuffix="" if key == "Score" else "%", margin=dict(t=50))
        figl.update_yaxes(autorange="reversed")
        st.plotly_chart(figl, width='stretch')

    st.subheader("Add ETFs to the backtest")
    pick_add = st.multiselect("ETFs to add to the Indian ETFs (NSE) market", list(t.Symbol), key="ia_add",
                              format_func=lambda x: f"{x} · {lst.set_index('Symbol').Name.get(x, '')}")
    if st.button("Add to backtest universe", disabled=not pick_add):
        prev = eng.PROFILE
        eng.set_profile("india")
        cmap_ = lst.set_index("Symbol")
        ents = {f"{x}.NS": ("India ETF", cmap_.Category.get(x, "Other Equity"), cmap_.Name.get(x, x)) for x in pick_add}
        with st.spinner("Adding..."):
            added, failed = eng.add_custom_tickers(ents)
        eng.set_profile(prev)
        get_prices.clear()
        st.session_state.pop(cat_key_for("india"), None)
        if added:
            st.success(f"Added {', '.join(added)} - pick Market = Indian ETFs (NSE) to use them.")
        if failed:
            st.error(f"No data for: {', '.join(failed)}")
    st.stop()

if view == "🌍 Global markets":
    st.subheader("Global markets comparison")
    st.caption("Country indices, Indian and US sectors, bond yields, currencies and commodities - "
               "independent of the Market chosen in the left panel.")
    cat_items = {sym: f"{eng.global_short(sym)} · {nm} ({grp})"
                 for grp, d in eng.GLOBAL_CATALOG.items() for sym, nm in d.items()}
    extra_key = "global_extra"
    if extra_key not in st.session_state:
        st.session_state[extra_key] = []
    for sym in st.session_state[extra_key]:
        cat_items.setdefault(sym, f"{sym} · custom")
    g1, g2 = st.columns([2, 1])
    preset = g1.selectbox("Preset", list(eng.GLOBAL_PRESETS), key="gl_preset")
    if g2.button("Load preset", width='stretch'):
        st.session_state["gl_sel"] = list(eng.GLOBAL_PRESETS[preset])
        st.rerun()
    if "gl_sel" not in st.session_state:
        st.session_state["gl_sel"] = list(eng.GLOBAL_PRESETS["Macro (like TradingView)"])
    with st.expander("Add whole groups or your own Yahoo symbol"):
        grp_pick = st.multiselect("Groups", list(eng.GLOBAL_CATALOG), key="gl_groups")
        if grp_pick and st.button("Add groups"):
            st.session_state["gl_sel"] = list(dict.fromkeys(
                st.session_state["gl_sel"] + [s_ for g in grp_pick for s_ in eng.GLOBAL_CATALOG[g]]))
            st.rerun()
        own = st.text_input("Yahoo symbols, comma-separated (e.g. ^NSEMDCP50, EWZ, ^MXX)", key="gl_own")
        if st.button("Add symbols", disabled=not own.strip()):
            new = [x.strip() for x in own.split(",") if x.strip()]
            st.session_state[extra_key] = list(dict.fromkeys(st.session_state[extra_key] + new))
            st.session_state["gl_sel"] = list(dict.fromkeys(st.session_state["gl_sel"] + new))
            st.rerun()
    sel_g = st.multiselect("Markets to compare", list(cat_items), key="gl_sel", format_func=lambda x: cat_items[x])
    h1, h2, h3, h4 = st.columns([2.2, 1.3, 1.6, 1])
    period = h1.radio("Period", ["1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Custom"],
                      index=0, horizontal=True, key="gl_period")
    base_opts = ["(none)"] + sel_g
    base = h2.selectbox("Reference", base_opts, index=base_opts.index("^NSEI") if "^NSEI" in sel_g else 0,
                        key="gl_base", format_func=lambda x: x if x == "(none)" else eng.global_short(x),
                        help="Drawn as a thick dashed grey line; used for the 'Relative' view.")
    gmode = h3.radio("Show", ["% change", "Relative to reference"], horizontal=True, key="gl_mode",
                     disabled=base == "(none)")
    if h4.button("⚡ Update", width='stretch', help="Fetch today's latest prices (Yahoo, ~15 min delayed)"):
        with st.spinner("Updating..."):
            try:
                eng.quick_update_global(sel_g)
            except Exception as e:
                st.error(f"Update failed: {e}")
        get_global.clear()
        st.session_state["last_update"] = datetime.now()
    if not sel_g:
        st.info("Pick at least one market.")
        st.stop()
    with st.spinner("Loading global prices (first time downloads each symbol once)..."):
        try:
            gclose = get_global(tuple(sorted(set(sel_g))), st.session_state.get("gl_token", 0))
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()
    missing = [x for x in sel_g if x not in gclose.columns]
    if missing:
        st.warning(f"No data found for: {', '.join(missing)}")
    cols_g = [x for x in sel_g if x in gclose.columns]
    if not cols_g:
        st.stop()
    g = gclose[cols_g].ffill()
    last = g.dropna(how="all").index[-1]
    offs = {"1W": pd.DateOffset(weeks=1), "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
            "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3),
            "5Y": pd.DateOffset(years=5)}
    if period == "YTD":
        sd = pd.Timestamp(year=last.year, month=1, day=1)
    elif period == "Custom":
        sd = pd.Timestamp(st.date_input("From", value=(last - pd.DateOffset(months=1)).date(),
                                        max_value=last.date(), key="gl_from"))
    else:
        sd = last - offs[period]
    prior = g.index[g.index <= sd]
    w = g.loc[(prior.max() if len(prior) else g.index[0]):last].dropna(axis=1, how="all")
    w = w[w.notna().any(axis=1)]
    if w.shape[0] < 2:
        st.warning("Not enough data for this period.")
        st.stop()
    chg = first_valid_norm(w)
    rel = gmode.startswith("Relative") and base in chg
    plotg = (((1 + chg / 100).div(1 + chg[base] / 100, axis=0) - 1) * 100) if rel else chg
    labg = lambda x: eng.global_short(x)
    fig, fin, colors = end_label_chart(plotg, base if base in plotg else None, 620, "%",
                                       "Return vs " + labg(base) + " (%)" if rel else "Change (%)",
                                       lambda v: f"{v:+.2f}%", 0.035, labg)
    fig.add_hline(y=0, line=dict(color="#aaaaaa", width=1, dash="dot"))
    st.plotly_chart(fig, width='stretch')
    st.caption(f"{w.index[0].date()} → {w.index[-1].date()} · each line starts at 0% · markets have different "
               "holidays, so missing days are carried forward · indices are price-only (no dividends) · "
               "yields show % change of the yield, FX pairs are quoted as shown (USDINR up = rupee weaker).")

    fs = pd.Series(fin).sort_values()
    bd = pd.DataFrame({"Market": [labg(x) for x in fs.index], "Return": fs.values})
    figb = px.bar(bd, x="Return", y="Market", orientation="h", text_auto=".2f",
                  color=bd["Return"] > 0, color_discrete_map={True: GREEN, False: RED},
                  title="Period change ranked" + (f" (vs {labg(base)})" if rel else ""))
    figb.update_layout(showlegend=False, height=max(320, 24 * len(bd)), xaxis_ticksuffix="%",
                       yaxis_title=None, margin=dict(t=50))
    st.plotly_chart(figb, width='stretch')

    st.subheader("Returns across periods (%)")
    hz = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756}
    rows = {}
    for x in cols_g:
        sr = gclose[x].dropna()
        rows[x] = {k: (sr.iloc[-1] / sr.iloc[-1 - n] - 1) * 100 if len(sr) > n else np.nan for k, n in hz.items()}
    tb = pd.DataFrame(rows).T
    if rel:
        tb = ((1 + tb / 100).div(1 + tb.loc[base] / 100, axis=1) - 1) * 100
    tb.insert(0, "Market", [eng.global_name(x) for x in tb.index])
    tb.index = [labg(x) for x in tb.index]
    st.dataframe(tb.style.format({k: "{:+.2f}" for k in hz}, na_rep="–")
                 .background_gradient(subset=list(hz), cmap="RdYlGn", vmin=-15, vmax=15),
                 width='stretch', height=min(40 + 35 * len(tb), 700))

    st.subheader("RSI trend")
    q1, q2, q3 = st.columns([1.2, 2, 2])
    rl = q1.selectbox("RSI length", [9, 14, 21, 50, 252], index=1, key="gl_rsilen")
    rof = q2.radio("RSI of", ["Price", "Ratio vs reference"], horizontal=True, key="gl_rsiof",
                   disabled=base == "(none)")
    sm = q3.slider("Smoothing (days)", 1, 10, 1, key="gl_rsism")
    srcg = g.copy()
    if rof.startswith("Ratio") and base in srcg:
        srcg = srcg.div(srcg[base], axis=0).drop(columns=[base])
    rs = eng.Indicators(srcg).rsi(int(rl))
    if sm > 1:
        rs = rs.rolling(sm, min_periods=1).mean()
    rs = rs.loc[w.index[0]:w.index[-1]].dropna(axis=1, how="all")
    if rs.empty:
        st.info("Not enough history for this RSI length.")
    else:
        figr, finr, _ = end_label_chart(rs, base if base in rs else None, 520, "", f"RSI({rl})",
                                        lambda v: f"{v:.1f}", 0.04, labg, yrange=[max(0, min(20, np.nanmin(rs.values) - 3)), 100],
                                        bands=[(70, 100, "rgba(192,57,43,0.10)"), (0, 30, "rgba(46,139,87,0.10)")])
        for lvl, dash in [(70, "dot"), (50, "dash"), (30, "dot")]:
            figr.add_hline(y=lvl, line=dict(color="#999999", width=1, dash=dash))
        st.plotly_chart(figr, width='stretch')
        nowr = pd.Series({labg(k): v for k, v in finr.items()}).sort_values(ascending=False)
        st.caption("Above 70 = overbought, below 30 = oversold, 50 = momentum line. "
                   + (f"**Overbought:** {', '.join(nowr[nowr >= 70].index)}. " if (nowr >= 70).any() else "")
                   + (f"**Oversold:** {', '.join(nowr[nowr <= 30].index)}. " if (nowr <= 30).any() else "")
                   + f"**Strongest:** {', '.join(nowr.index[:3])}.")
    st.stop()

if view == "📈 Sector comparison":
    bench = eng.BENCHMARK
    avail = [t for t in eng.UNIVERSE if t in close.columns]
    cat_of = lambda t: eng.CATEGORY.get(t, t)
    reps = {}
    for t in avail:                                   # default: one ETF per category (benchmark shown anyway)
        if t != bench:
            reps.setdefault(cat_of(t), t)
    st.subheader(f"Performance vs {bench}")
    c1, c2, c3 = st.columns([2.2, 1.6, 1.2])
    period = c1.radio("Period", ["1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Custom"],
                      index=2, horizontal=True, key=f"sc_period_{market}")
    mode = c2.radio("Show", ["% change", f"Relative to {short(bench)}"], horizontal=True,
                    key=f"sc_mode_{market}",
                    help="Relative = how much each ETF beat (+) or lagged (-) the benchmark since the start date.")
    by_cat = c3.toggle("Average by category", value=False, key=f"sc_bycat_{market}")
    last = close.index[-1]
    offsets = {"1W": pd.DateOffset(weeks=1), "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
               "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3),
               "5Y": pd.DateOffset(years=5)}
    if period == "YTD":
        start_d = pd.Timestamp(year=last.year, month=1, day=1)
    elif period == "Custom":
        start_d = pd.Timestamp(st.date_input("From", value=(last - pd.DateOffset(months=3)).date(),
                                             max_value=last.date(), key=f"sc_from_{market}"))
    else:
        start_d = last - offsets[period]
    picks = st.multiselect("ETFs to compare", avail, default=list(reps.values()), key=f"sc_sel_{market}",
                           format_func=lambda t: f"{t} · {eng.NAMES.get(t, cat_of(t))}")
    tickers = list(dict.fromkeys([bench] + picks))
    win = close.loc[close.index >= start_d - pd.Timedelta(days=5), [t for t in tickers if t in close.columns]]
    win = win.loc[win.index >= win.index[win.index <= start_d].max()] if (win.index <= start_d).any() else win
    win = win.ffill().dropna(axis=1, how="all")
    if win.shape[0] < 2 or bench not in win:
        st.warning("Not enough data for this period.")
        st.stop()
    pct_chg = first_valid_norm(win)
    if mode.startswith("Relative"):
        b = pct_chg[bench]
        plot_df = ((1 + pct_chg.div(100)).div(1 + b.div(100), axis=0) - 1) * 100
        ylab = f"Return vs {short(bench)} (%)"
    else:
        plot_df = pct_chg
        ylab = "Change (%)"
    if by_cat:
        others = [t for t in plot_df.columns if t != bench]
        grp = plot_df[others].T.groupby(lambda t: cat_of(t)).mean().T
        plot_df = pd.concat([plot_df[[bench]], grp], axis=1)
    label = (lambda col: short(col) if col in eng.UNIVERSE or col == bench else col)

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    fig = go.Figure()
    finals = {}
    for i, col in enumerate(plot_df.columns):
        ser = plot_df[col].dropna()
        if ser.empty:
            continue
        is_b = col == bench
        color = "#888888" if is_b else palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=ser.index, y=ser, mode="lines", name=label(col),
            line=dict(color=color, width=3.5 if is_b else 1.8, dash="dash" if is_b else "solid"),
            hovertemplate=f"<b>{label(col)}</b> %{{x|%d %b %Y}}: %{{y:.2f}}%<extra></extra>"))
        finals[col] = (ser.iloc[-1], color)
    rng_y = max(1.0, plot_df.max().max() - plot_df.min().min())
    pos = spread_labels([(k, v[0]) for k, v in finals.items()], rng_y * 0.035)
    for col, (v, color) in finals.items():
        fig.add_annotation(x=plot_df.index[-1], y=pos[col], text=f"<b>{label(col)}</b> {v:+.2f}%",
                           xanchor="left", xshift=8, showarrow=False, font=dict(color="white", size=11),
                           bgcolor=color, borderpad=2)
    fig.add_hline(y=0, line=dict(color="#aaaaaa", width=1, dash="dot"))
    fig.update_layout(height=620, hovermode="closest", showlegend=False, yaxis_ticksuffix="%",
                      yaxis_title=ylab, margin=dict(t=30, r=190, b=10), xaxis=dict(range=[plot_df.index[0], plot_df.index[-1]]))
    st.plotly_chart(fig, width='stretch')
    st.caption(f"From {plot_df.index[0].date()} to {plot_df.index[-1].date()} · each line starts at 0% "
               f"(ETFs launched later start from their first price) · dashed grey = {bench}"
               + (" · prices in USD" if eng.TO_USD else ""))

    # ---- RSI trend chart for the same ETFs and period
    st.subheader("RSI trend")
    r1, r2, r3 = st.columns([1.2, 2, 2])
    rsi_len = r1.selectbox("RSI length", [9, 14, 21, 50, 252], index=1, key=f"sc_rsilen_{market}")
    rsi_of = r2.radio("RSI of", ["Price", f"Ratio vs {short(bench)}"], horizontal=True, key=f"sc_rsiof_{market}",
                      help="Ratio = RSI of ETF price ÷ benchmark price. Above 50 means the ETF is gaining "
                           "strength against the benchmark.")
    smooth = r3.slider("Smoothing (days)", 1, 10, 1, key=f"sc_rsism_{market}",
                       help="Moving average of the RSI line to reduce noise (1 = none).")
    cols_r = [t for t in tickers if t in close.columns]
    src = close[cols_r].ffill()
    if rsi_of.startswith("Ratio"):
        src = src.div(close[bench].ffill(), axis=0).drop(columns=[bench])
    rsi_full = eng.Indicators(src).rsi(int(rsi_len))
    if smooth > 1:
        rsi_full = rsi_full.rolling(smooth, min_periods=1).mean()
    rsi = rsi_full.loc[plot_df.index[0]:plot_df.index[-1]].dropna(axis=1, how="all")
    if by_cat:
        oth = [t for t in rsi.columns if t != bench]
        rsi = pd.concat([rsi[[bench]] if bench in rsi else None,
                         rsi[oth].T.groupby(lambda t: cat_of(t)).mean().T], axis=1)
    if rsi.empty:
        st.info("Not enough history to compute RSI for this period - pick a longer period or shorter RSI length.")
    else:
        color_of = {col: finals[col][1] for col in finals}
        figr = go.Figure()
        figr.add_hrect(y0=70, y1=100, fillcolor="rgba(192,57,43,0.10)", line_width=0)
        figr.add_hrect(y0=0, y1=30, fillcolor="rgba(46,139,87,0.10)", line_width=0)
        for lvl, dash in [(70, "dot"), (50, "dash"), (30, "dot")]:
            figr.add_hline(y=lvl, line=dict(color="#999999", width=1, dash=dash))
        rf = {}
        for i, col in enumerate(rsi.columns):
            ser = rsi[col].dropna()
            if ser.empty:
                continue
            is_b = col == bench
            color = color_of.get(col, palette[i % len(palette)])
            figr.add_trace(go.Scatter(
                x=ser.index, y=ser, mode="lines", name=label(col),
                line=dict(color=color, width=3 if is_b else 1.6, dash="dash" if is_b else "solid"),
                hovertemplate=f"<b>{label(col)}</b> %{{x|%d %b %Y}}: RSI %{{y:.1f}}<extra></extra>"))
            rf[col] = (ser.iloc[-1], color)
        pos_r = spread_labels([(k, v[0]) for k, v in rf.items()], 3.2)
        for col, (v, color) in rf.items():
            figr.add_annotation(x=rsi.index[-1], y=pos_r[col], text=f"<b>{label(col)}</b> {v:.1f}",
                                xanchor="left", xshift=8, showarrow=False, font=dict(color="white", size=11),
                                bgcolor=color, borderpad=2)
        lo = min(20, np.nanmin(rsi.values) - 3)
        hi = max(80, max(pos_r.values()) + 3 if pos_r else 80)
        figr.update_layout(height=520, showlegend=False, hovermode="closest", yaxis=dict(range=[lo, hi], title=f"RSI({rsi_len})"),
                           margin=dict(t=20, r=190, b=10), xaxis=dict(range=[rsi.index[0], rsi.index[-1]]))
        st.plotly_chart(figr, width='stretch')
        now = pd.Series({label(k): v[0] for k, v in rf.items()}).sort_values(ascending=False)
        ob, os_ = now[now >= 70], now[now <= 30]
        st.caption(f"RSI({rsi_len}) of {'price' if rsi_of == 'Price' else 'ETF ÷ ' + short(bench)} · "
                   "shaded red above 70 = overbought, green below 30 = oversold, dashed 50 = momentum line. "
                   + (f"**Overbought now:** {', '.join(ob.index)}. " if len(ob) else "")
                   + (f"**Oversold now:** {', '.join(os_.index)}. " if len(os_) else "")
                   + f"**Strongest:** {', '.join(now.index[:3])}.")

    # ranked bars for the period
    fin = plot_df.ffill().iloc[-1].drop(bench, errors="ignore").sort_values()
    bdf = pd.DataFrame({"Item": [label(t) for t in fin.index], "Return": fin.values})
    figb = px.bar(bdf, x="Return", y="Item", orientation="h", text_auto=".2f",
                  color=bdf["Return"] > (0 if mode.startswith("Relative") else plot_df[bench].iloc[-1]),
                  color_discrete_map={True: GREEN, False: RED},
                  title=("Beat / lagged the benchmark" if mode.startswith("Relative")
                         else f"Period return (green = beat {short(bench)} {plot_df[bench].iloc[-1]:+.2f}%)"))
    if not mode.startswith("Relative"):
        figb.add_vline(x=plot_df[bench].iloc[-1], line=dict(color="#888888", dash="dash"))
    figb.update_layout(showlegend=False, height=max(320, 24 * len(bdf)), xaxis_ticksuffix="%",
                       yaxis_title=None, margin=dict(t=50))
    st.plotly_chart(figb, width='stretch')

    # multi-period table
    st.subheader("Returns across periods (%)")
    hz = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756}
    rows = {}
    for t in tickers:
        if t not in close.columns:
            continue
        sr = close[t].dropna()
        rows[t] = {k: (sr.iloc[-1] / sr.iloc[-1 - n] - 1) * 100 if len(sr) > n else np.nan for k, n in hz.items()}
    tbl = pd.DataFrame(rows).T
    if mode.startswith("Relative"):
        bt = tbl.loc[bench]
        tbl = ((1 + tbl / 100).div(1 + bt / 100, axis=1) - 1) * 100
    tbl.insert(0, "Category", [cat_of(t) if t != bench else "Benchmark" for t in tbl.index])
    tbl.insert(0, "Name", [eng.NAMES.get(t, "") for t in tbl.index])
    num = list(hz)
    st.dataframe(tbl.style.format({k: "{:+.2f}" for k in num}, na_rep="–")
                 .background_gradient(subset=num, cmap="RdYlGn", vmin=-15, vmax=15),
                 width='stretch', height=min(40 + 35 * len(tbl), 700))
    st.stop()

runs = st.session_state.runs
if not runs:
    st.info("Set your parameters in the left panel and click **▶ Run backtest**. "
            "Defaults match your final Momentify configuration.")
    st.stop()

names = list(runs)
cur = st.session_state.get("current", names[-1])
sel = st.selectbox("Showing run", names, index=names.index(cur) if cur in names else len(names) - 1)
R = runs[sel]
c, eq, trades, s = R["cfg"], R["eq"], R["trades"], R["stats"]

st.caption(
    f"{eq.index[0].date()} → {eq.index[-1].date()} · top {c.n_holdings}, exit rank {c.exit_rank} · "
    f"rank={c.rank_criteria} {c.periods}/{c.weights} · {c.rebalance} · "
    f"entry: {c.entry_condition} · exit: {c.exit_condition} · "
    f"stop={'off' if not c.stop_loss else f'{c.stop_loss:.0%}'} · RS={c.rs_rule if c.use_rs else 'off'} · "
    f"retracement={'off' if c.retracement is None else f'{c.retracement:.0%}'} · {c.cost_bps:g} bps/side"
    + (f" · ETF group: max {c.max_per_category}/category (buffer {c.category_buffer})" if c.max_per_category else " · ETF group: off")
    + (f" · excluded: {', '.join(c.exclude)}" if c.exclude else "")
    + (f" · ETFs tested ({len(c.include)}): {', '.join(c.include)}" if c.include else " · all ETFs"))
cmap = eng.category_map(c)

tab_over, tab_ret, tab_trd, tab_reb, tab_sig, tab_cmp = st.tabs(
    ["Overview", "Returns", "Trades", "🔄 Rebalance changes", "Today's signal", "Compare runs"])

# ---------------- Overview
with tab_over:
    spy = R["bm_stats"].get(eng.bench_label(), {})
    k = st.columns(6)
    k[0].metric("CAGR", pct(s["CAGR"]), f"{(s['CAGR'] - spy.get('CAGR', np.nan)) * 100:+.2f} pts vs {eng.BENCHMARK}" if spy else None)
    k[1].metric("Max drawdown", pct(s["MaxDD"]), f"{eng.BENCHMARK} {pct(spy.get('MaxDD'))}" if spy else None, delta_color="off")
    k[2].metric("Sharpe", f"{s['Sharpe']:.2f}", f"{eng.BENCHMARK} {spy.get('Sharpe', np.nan):.2f}" if spy else None, delta_color="off")
    k[3].metric("Final value", f"{eng.CCY}{eq['Equity'].iloc[-1]:,.0f}", f"from {eng.CCY}{c.capital:,.0f}", delta_color="off")
    wins = (trades.PnL > 0).mean() if len(trades) else np.nan
    k[4].metric("Trades", f"{len(trades)}", f"win rate {pct(wins, 1)}", delta_color="off")
    k[5].metric("Longest drawdown", f"{s['LongestDrawdownDays']} days", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq["Equity"], name="Strategy", line=dict(color=BLUE, width=2)))
    for (name, ser), col in zip(R["bm"].items(), [GREY, GOLD]):
        fig.add_trace(go.Scatter(x=ser.index, y=ser, name=name, line=dict(color=col, width=1.2)))
    log = st.toggle("Log scale", value=True)
    fig.update_layout(title="Equity curve", height=430, yaxis_type="log" if log else "linear",
                      yaxis_tickprefix=eng.CCY, hovermode="x unified", margin=dict(t=50, b=10),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width='stretch')

    dd = eq["Equity"] / eq["Equity"].cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", name="Strategy",
                             line=dict(color=RED, width=1)))
    if eng.bench_label() in R["bm"]:
        b = R["bm"][eng.bench_label()]
        fig.add_trace(go.Scatter(x=b.index, y=(b / b.cummax() - 1) * 100, name=eng.BENCHMARK,
                                 line=dict(color=GREY, width=1, dash="dot")))
    fig.update_layout(title="Drawdown (%)", height=250, hovermode="x unified",
                      yaxis_ticksuffix="%", margin=dict(t=50, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width='stretch')

    rows = [("Strategy", s)] + list(R["bm_stats"].items())
    tbl = pd.DataFrame({n: {"CAGR": pct(x["CAGR"]), "Volatility": pct(x["Volatility"]),
                            "Sharpe": f"{x['Sharpe']:.2f}", "Sortino": f"{x['Sortino']:.2f}",
                            "Max DD": pct(x["MaxDD"]), "Calmar": f"{x['Calmar']:.2f}"} for n, x in rows}).T
    st.subheader("Strategy vs benchmarks")
    st.dataframe(tbl, width='stretch')

# ---------------- Returns
with tab_ret:
    mt = eng.monthly_table(eq["Equity"])
    yr = mt.pop("Year Total")
    st.subheader("Monthly returns (%)")
    fig = px.imshow(mt, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, zmin=-10, zmax=10)
    fig.update_layout(height=max(300, 26 * len(mt)), coloraxis_showscale=False, margin=dict(t=10))
    fig.update_yaxes(dtick=1)
    st.plotly_chart(fig, width='stretch')

    st.subheader(f"Yearly returns vs {eng.BENCHMARK} (%)")
    yd = pd.DataFrame({"Strategy": yr})
    if eng.bench_label() in R["bm"]:
        yd[eng.BENCHMARK] = eng.monthly_table(R["bm"][eng.bench_label()])["Year Total"]
    yl = yd.reset_index().melt(id_vars="Year", var_name="Series", value_name="Return")
    fig = px.bar(yl, x="Year", y="Return", color="Series", barmode="group",
                 color_discrete_map={"Strategy": BLUE, eng.BENCHMARK: GREY})
    fig.update_layout(height=380, yaxis_ticksuffix="%", margin=dict(t=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    hold = eq["Holdings"].str.split(",").explode()
    hold = hold[hold.str.len() > 0]
    if len(hold):
        by = st.radio("Group holdings by", ["Asset class", "Category"], horizontal=True)
        exp = hold.map(eng.UNIVERSE if by == "Asset class" else cmap).groupby(level=0).value_counts().unstack(fill_value=0)
        exp = exp.resample("ME").mean()
        cash_slots = (c.n_holdings - exp.sum(axis=1)).clip(lower=0)
        exp[f"Cash ({eng.CASH_ETF})"] = cash_slots
        st.subheader(f"Average slots by {by.lower()} (monthly)")
        fig = px.area(exp, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=330, yaxis_title="Slots", margin=dict(t=10), legend_title=None,
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width='stretch')

# ---------------- Trades
with tab_trd:
    if trades.empty:
        st.warning("No trades in this run.")
    else:
        c1, c2, c3 = st.columns(3)
        gb = c1.radio("P&L by", ["Category", "Asset class"], horizontal=True, key="pnl_by")
        g = trades.groupby("Category" if gb == "Category" and "Category" in trades else "Group")["PnL"].agg(["count", "sum"]).rename(columns={"count": "Trades", "sum": "Net P&L"})
        fig = px.bar(g.sort_values("Net P&L"), x="Net P&L", orientation="h", text_auto=",.0f",
                     color=g.sort_values("Net P&L")["Net P&L"] > 0,
                     color_discrete_map={True: GREEN, False: RED}, title=f"Net P&L by {gb.lower()} ({eng.CCY})")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40))
        c1.plotly_chart(fig, width='stretch')

        r = trades.groupby("ExitReason")["PnL"].agg(["count", "sum"]).rename(columns={"count": "Trades", "sum": "Net P&L"})
        fig = px.bar(r.sort_values("Net P&L"), x="Net P&L", orientation="h", text_auto=",.0f",
                     color=r.sort_values("Net P&L")["Net P&L"] > 0,
                     color_discrete_map={True: GREEN, False: RED}, title=f"Net P&L by exit reason ({eng.CCY})")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40))
        c2.plotly_chart(fig, width='stretch')

        t = trades.groupby("Ticker")["PnL"].sum().sort_values()
        t = pd.concat([t.head(6), t.tail(6)]).drop_duplicates()
        fig = px.bar(t, orientation="h", text_auto=",.0f", color=t > 0,
                     color_discrete_map={True: GREEN, False: RED}, title=f"Best & worst ETFs ({eng.CCY})")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40), yaxis_title=None, xaxis_title="Net P&L")
        c3.plotly_chart(fig, width='stretch')

        wins, losses = trades[trades.PnL > 0], trades[trades.PnL <= 0]
        m = st.columns(5)
        m[0].metric("Win rate", pct(len(wins) / len(trades), 1))
        m[1].metric("Avg win", f"{eng.CCY}{wins.PnL.mean():,.0f}" if len(wins) else "–")
        m[2].metric("Avg loss", f"{eng.CCY}{losses.PnL.mean():,.0f}" if len(losses) else "–")
        m[3].metric("Avg hold", f"{trades.DaysHeld.mean():.0f} days")
        m[4].metric("Trades / month", f"{len(trades) / max(s['Years'] * 12, 1):.1f}")

        st.subheader("Trade log")
        f1, f2 = st.columns(2)
        gsel = f1.multiselect("Filter group", sorted(trades.Group.unique()))
        rsel = f2.multiselect("Filter exit reason", sorted(trades.ExitReason.unique()))
        view = trades
        if gsel:
            view = view[view.Group.isin(gsel)]
        if rsel:
            view = view[view.ExitReason.isin(rsel)]
        st.dataframe(view.sort_values("ExitDate", ascending=False), width='stretch', height=380,
                     column_config={"PnL": st.column_config.NumberColumn(format=f"{eng.CCY}%.0f"),
                                    "ReturnPct": st.column_config.NumberColumn("Return %", format="%.2f%%")})
        st.download_button("Download trades CSV", view.to_csv(index=False), f"trades_{sel}.csv", "text/csv")

# ---------------- Rebalance changes (incoming / outgoing per rebalance)
with tab_reb:
    def money(v):
        return f"{eng.CCY}{v:,.2f}"

    def nm(t):
        return eng.NAMES.get(t, cmap.get(t, ""))

    def styled(df, header_color):
        return (df.style.format({"Allocated Amt": money, "Price": "{:,.2f}", "Qty": "{:,.2f}"})
                .set_table_styles([{"selector": "th", "props": [("background-color", header_color),
                                                                 ("color", "white"), ("font-weight", "bold")]}]))

    tr = trades.copy()
    if tr.empty:
        st.info("No trades in this run.")
    else:
        tr["EntryDate"] = pd.to_datetime(tr["EntryDate"])
        tr["ExitDate"] = pd.to_datetime(tr["ExitDate"])
        is_eop = tr.ExitReason == "End of Period"
        events = sorted(set(tr.EntryDate) | set(tr.loc[~is_eop, "ExitDate"]), reverse=True)
        ev = st.selectbox("Rebalance / trade date", events, format_func=lambda d_: d_.strftime("%a %d %b %Y"),
                          help="Latest first. Pick any earlier date to see what changed then.")
        inc = tr[tr.EntryDate == ev]
        out = tr[(tr.ExitDate == ev) & ~is_eop]
        before = tr[(tr.EntryDate < ev) & (tr.ExitDate >= ev)]
        after = tr[(tr.EntryDate <= ev) & ((tr.ExitDate > ev) | is_eop)]
        k = st.columns(4)
        k[0].metric("Incoming", len(inc))
        k[1].metric("Outgoing", len(out))
        k[2].metric("Held before", len(before))
        k[3].metric("Held after", len(after))

        st.markdown("#### 🟢 Incoming ETFs")
        if inc.empty:
            st.caption("Nothing bought on this date.")
        else:
            t_in = pd.DataFrame({"Symbol": inc.Ticker, "Name": [nm(t) for t in inc.Ticker],
                                 "Allocated Amt": inc.EntryPrice * inc.Shares, "Price": inc.EntryPrice,
                                 "Qty": inc.Shares}).reset_index(drop=True)
            t_in.index = t_in.index + 1
            st.dataframe(styled(t_in, "#1abc9c"), width='stretch')

        st.markdown("#### 🔴 Outgoing ETFs")
        if out.empty:
            st.caption("Nothing sold on this date.")
        else:
            t_out = pd.DataFrame({"Symbol": out.Ticker, "Name": [nm(t) for t in out.Ticker],
                                  "Allocated Amt": out.ExitPrice * out.Shares, "Price": out.ExitPrice,
                                  "Qty": out.Shares, "Reason": out.ExitReason,
                                  "P&L": out.PnL, "Return %": out.ReturnPct,
                                  "Held since": out.EntryDate.dt.strftime("%d %b %Y")}).reset_index(drop=True)
            t_out.index = t_out.index + 1
            st.dataframe(styled(t_out, "#e74c3c").format({"P&L": money, "Return %": "{:+.2f}%"},
                                                         subset=["P&L", "Return %"]), width='stretch')

        c1, c2 = st.columns(2)
        c1.markdown("#### Holdings before")
        c1.dataframe(pd.DataFrame({"Symbol": before.Ticker, "Category": [cmap.get(t, "") for t in before.Ticker],
                                   "Since": before.EntryDate.dt.strftime("%d %b %Y")}).reset_index(drop=True),
                     width='stretch', hide_index=True)
        c2.markdown("#### Holdings after")
        c2.dataframe(pd.DataFrame({"Symbol": after.Ticker, "Category": [cmap.get(t, "") for t in after.Ticker],
                                   "Status": ["New" if d_ == ev else "Kept" for d_ in after.EntryDate]})
                     .reset_index(drop=True), width='stretch', hide_index=True)
        st.caption("Allocated Amt = Price × Qty at the trade price (next day's close after the signal). "
                   "Quantities are fractional in the backtest; round down to whole units when placing real orders.")

        # ---- preview of the next rebalance on the latest prices
        st.divider()
        st.markdown("### 🔮 Next rebalance preview (on latest prices)")
        held_now = list(tr.loc[is_eop, "Ticker"])
        try:
            d_prev, sells, buys, keep = eng.next_rebalance_preview(close, c, held_now)
            eq_now = eq["Equity"].iloc[-1]
            slot_amt = eq_now / c.n_holdings
            lastp = close.ffill().iloc[-1]
            st.caption(f"Based on prices up to {d_prev.date()} · current holdings: "
                       f"{', '.join(held_now) if held_now else 'none'} · use ⚡ Update to latest prices first.")
            p1, p2 = st.columns(2)
            p1.markdown("#### 🟢 Would buy")
            if buys:
                pb = pd.DataFrame({"Symbol": buys, "Name": [nm(t) for t in buys],
                                   "Allocated Amt": [slot_amt] * len(buys), "Price": [lastp[t] for t in buys],
                                   "Qty": [slot_amt / lastp[t] for t in buys]})
                pb.index = pb.index + 1
                p1.dataframe(styled(pb, "#1abc9c"), width='stretch')
            else:
                p1.caption("No new buys.")
            p2.markdown("#### 🔴 Would sell")
            if sells:
                ps = pd.DataFrame({"Symbol": [t for t, _ in sells], "Reason": [r_ for _, r_ in sells],
                                   "Price": [lastp.get(t, np.nan) for t, _ in sells]})
                ps.index = ps.index + 1
                p2.dataframe(ps.style.format({"Price": "{:,.2f}"}), width='stretch')
            else:
                p2.caption("No sells - all holdings still qualify.")
            if keep:
                st.caption(f"Kept: {', '.join(keep)}. "
                           + (f"Empty slots → {eng.CASH_ETF}." if len(keep) + len(buys) < c.n_holdings else ""))
        except Exception as e_:
            st.warning(f"Preview not available: {e_}")

# ---------------- Today's signal
with tab_sig:
    pxs, score, rank, elig, xflag = eng.compute_signals(close, c)
    d = pxs.index[-1]
    sig = pd.DataFrame({"Name": [eng.NAMES.get(t, "") for t in pxs.columns],
                        "Group": [eng.UNIVERSE[t] for t in pxs.columns],
                        "Category": [cmap.get(t, t) for t in pxs.columns], "Score": score.loc[d],
                        "Rank": rank.loc[d], "Close": pxs.loc[d],
                        "Exit signal": xflag.loc[d], "Eligible": elig.loc[d]}).sort_values("Rank")
    picks = eng.pick_candidates(score.loc[d], elig.loc[d], {}, pxs.loc[d], c.n_holdings, c, cmap)
    buys = sig.loc[picks]
    if c.max_per_category:
        best_in_cat = sig[sig.Eligible].groupby("Category").head(c.max_per_category).index
        sig["Best in category"] = sig.index.isin(best_in_cat)
    st.subheader(f"Signal as of {d.date()}")
    cols = st.columns(max(c.n_holdings, 1))
    for i in range(c.n_holdings):
        if i < len(buys):
            t = buys.index[i]
            cols[i].metric(f"#{int(buys.Rank.iloc[i])} · {buys.Category.iloc[i]}", t, f"score {buys.Score.iloc[i]:.2f}",
                           delta_color="off")
        else:
            cols[i].metric("Empty slot", eng.CASH_ETF if c.use_cash_etf else "Cash", delta_color="off")
    if not eq.empty:
        st.caption(f"Backtest holdings on its last day: {eq['Holdings'].iloc[-1] or 'none'} "
                   "(differences come from the exit-rank buffer: existing holdings are kept while rank ≤ exit rank).")
    # ---- add the cash ETF as a reference row (it is never "bought" through ranking)
    cs = eng.cash_series(close)
    cash_note = ""
    if cs is not None and cs.notna().sum() > max(c.periods):
        cscore = eng.compute_score(cs.to_frame(eng.CASH_ETF), c)[eng.CASH_ETF].reindex(score.index).ffill().loc[d]
        if pd.notna(cscore):
            above = int((score.loc[d] > cscore).sum())
            crow = pd.DataFrame({"Name": [eng.NAMES.get(eng.CASH_ETF, "Cash / liquid ETF (empty slots)")],
                                 "Group": ["Cash"], "Category": ["Cash"], "Score": [cscore],
                                 "Rank": [above + 0.5], "Close": [cs.ffill().loc[:d].iloc[-1]],
                                 "Exit signal": [False], "Eligible": [False]}, index=[f"💵 {eng.CASH_ETF}"])
            if "Best in category" in sig:
                crow["Best in category"] = False
            sig = pd.concat([sig, crow]).sort_values("Rank")
            pos_txt = ("would rank above every ETF (#1)" if above == 0 else
                       f"would rank between #{above} and #{above + 1}")
            cash_note = (f"💵 **{eng.CASH_ETF}** (cash) scores **{cscore:.2f}** and {pos_txt}. "
                         f"{above} ETF(s) score higher than cash"
                         + (" - with Sharpe ranking a liquid fund can score very high because its price barely "
                            "moves (tiny volatility), so this mainly shows how many ETFs have a steadier trend "
                            "than cash." if c.rank_criteria == "sharpe" else
                            "; ETFs below this line are earning less than simply holding cash.")
                         + " The cash ETF is not part of the ranking: it only fills empty slots.")
    else:
        cash_note = f"{eng.CASH_ETF}: not enough price history to score it."
    st.caption(cash_note)

    def _hl(row):
        return ["background-color: rgba(201,162,39,0.25)" if str(row.name).startswith("💵") else ""] * len(row)
    st.dataframe(sig.round(3).style.apply(_hl, axis=1).format({"Rank": lambda r: f"{r:g}"}, precision=3),
                 width='stretch', height=560)

# ---------------- Compare runs
with tab_cmp:
    st.write("Every run you make is kept here for this session. Run the standard set to test the open questions in one go.")
    if st.button("Run standard comparison set (16 variants, uses current sidebar settings as base)"):
        variants = [
            ("Base", {}), ("Monthly", dict(rebalance="monthly")), ("Biweekly", dict(rebalance="biweekly")),
            ("No stop", dict(stop_loss=None)), ("Stop 8%", dict(stop_loss=0.08)),
            ("Stop 10%", dict(stop_loss=0.10)), ("Stop 15%", dict(stop_loss=0.15)),
            ("No RS filter", dict(use_rs=False)),
            ("Ex gold & silver", dict(exclude=tuple(set(cfg.exclude) | set(eng.GOLD_SILVER)))),
            ("Rank by return", dict(rank_criteria="return")),
            ("Monthly, no RS", dict(rebalance="monthly", use_rs=False)),
            ("ETF Group on (1/category)", dict(max_per_category=1)),
            ("ETF Group off", dict(max_per_category=0)),
            ("Entry: Golden Cross state", dict(entry_condition="Golden Cross state (50 EMA > 200 EMA)")),
            ("Entry: Dual Momentum", dict(entry_condition="Dual Momentum (12M return > T-bills)")),
            ("Exit: rank only", dict(exit_condition="None (rank exit only)")),
        ]
        bar = st.progress(0.0)
        for i, (nm, kw) in enumerate(variants):
            name = f"Std · {nm}"
            st.session_state.runs[name] = run(replace(cfg, name=name, **kw), close)
            bar.progress((i + 1) / len(variants), text=name)
        st.rerun()

    picked = st.multiselect("Runs to compare", list(runs), default=list(runs)[-min(len(runs), 6):])
    if picked:
        rows = []
        for n in picked:
            x = runs[n]
            t = x["trades"]
            rows.append(dict(Run=n, CAGR=x["stats"]["CAGR"] * 100, Vol=x["stats"]["Volatility"] * 100,
                             Sharpe=x["stats"]["Sharpe"], Sortino=x["stats"]["Sortino"],
                             MaxDD=x["stats"]["MaxDD"] * 100, Calmar=x["stats"]["Calmar"],
                             Trades=len(t), WinRate=(t.PnL > 0).mean() * 100 if len(t) else np.nan))
        cmp = pd.DataFrame(rows).set_index("Run")
        best = cmp["Sharpe"].idxmax()
        st.success(f"Best Sharpe: **{best}** · CAGR {cmp.loc[best, 'CAGR']:.2f}% · Max DD {cmp.loc[best, 'MaxDD']:.2f}%")
        st.dataframe(cmp.style.format({"CAGR": "{:.2f}%", "Vol": "{:.2f}%", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                                       "MaxDD": "{:.2f}%", "Calmar": "{:.2f}", "WinRate": "{:.1f}%"})
                     .background_gradient(subset=["Sharpe", "CAGR"], cmap="Greens"),
                     width='stretch')

        fig = px.scatter(cmp.reset_index(), x="MaxDD", y="CAGR", text="Run", size=[12] * len(cmp),
                         title="Return vs drawdown (up and to the right is better)")
        fig.update_traces(textposition="top center", marker_color=BLUE)
        fig.update_layout(height=420, xaxis_ticksuffix="%", yaxis_ticksuffix="%")
        st.plotly_chart(fig, width='stretch')

        fig = go.Figure()
        for n in picked:
            e = runs[n]["eq"]["Equity"]
            fig.add_trace(go.Scatter(x=e.index, y=e / e.iloc[0] * 100, name=n, mode="lines"))
        first = runs[picked[0]]
        if eng.bench_label() in first["bm"]:
            b = first["bm"][eng.bench_label()]
            fig.add_trace(go.Scatter(x=b.index, y=b / b.iloc[0] * 100, name=eng.BENCHMARK,
                                     line=dict(color="black", dash="dot", width=1)))
        fig.update_layout(title="Growth of 100 (log)", yaxis_type="log", height=450, hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
        st.download_button("Download comparison CSV", cmp.to_csv(), "comparison.csv", "text/csv")

    if st.button("Clear all runs"):
        st.session_state.runs_by_market[market] = {}
        st.rerun()

st.caption("Analytical tool only — not investment advice. Prices: Yahoo Finance adjusted closes. "
           "Trades execute at the close of the day after each signal.")
