"""
US ETF Momentum Rotation - Backtester
=====================================
Port of the Momentify "Virtual ETF Strategy Claude" rules to US ETFs.

Rules (defaults = your final Momentify configuration):
  * Universe     : ~32 liquid US ETFs, one per exposure (see UNIVERSE below)
  * Ranking      : Sharpe return, blended over 252 / 90 / 60 days, weights 2 / 1 / 1
  * Entry filters: close > 200 EMA, close within 20% of 252-day high,
                   relative strength (ETF / SPY ratio) above its own 200 EMA
  * Holdings     : top 5, equal weight; hold while rank <= 10
  * Exits        : rank > 10, close below 200 EMA, or 12% stop loss
  * Rebalance    : weekly (Wednesday) - also monthly / biweekly
  * Empty slots  : parked in BIL (T-bills) instead of idle cash
  * Execution    : signals on day-T close, trades at day-T+1 close, with costs

Usage examples:
  python us_etf_momentum.py                       # base run
  python us_etf_momentum.py --rebalance monthly   # monthly rebalance
  python us_etf_momentum.py --stop 0.10 --no-rs   # change parameters
  python us_etf_momentum.py --exclude GLD,SLV     # drop tickers
  python us_etf_momentum.py --sweep               # run the comparison set
  python us_etf_momentum.py --refresh             # re-download prices

Analytical tool only - not investment advice.
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Universe: ticker -> group (one ETF per exposure)
# --------------------------------------------------------------------------
UNIVERSE = {
    # US equity
    "SPY": "US Equity", "QQQ": "US Equity", "IWM": "US Equity", "MDY": "US Equity",
    # Sectors
    "XLK": "Sector", "XLF": "Sector", "XLE": "Sector", "XLV": "Sector",
    "XLI": "Sector", "XLY": "Sector", "XLP": "Sector", "XLU": "Sector",
    "XLB": "Sector", "XLRE": "Sector", "XLC": "Sector", "SMH": "Sector",
    # International
    "EFA": "International", "EEM": "International", "EWJ": "International",
    "INDA": "International", "EWZ": "International", "FXI": "International",
    "EWG": "International",
    # Bonds
    "TLT": "Bonds", "IEF": "Bonds", "LQD": "Bonds", "HYG": "Bonds", "TIP": "Bonds",
    # Commodities
    "GLD": "Commodity", "SLV": "Commodity", "DBC": "Commodity", "USO": "Commodity",
}
# ETF Group categories: ETFs that move together share a category. With the ETF Group
# option on, at most `max_per_category` ETFs (default 1 = the best one) are held per category.
CATEGORY = {
    "SPY": "US Broad Market", "QQQ": "US Broad Market", "IWM": "US Broad Market", "MDY": "US Broad Market",
    "XLK": "Technology", "SMH": "Technology", "XLC": "Technology",
    "XLE": "Energy & Oil", "USO": "Energy & Oil",
    "XLF": "Cyclicals", "XLI": "Cyclicals", "XLB": "Cyclicals", "XLY": "Cyclicals",
    "XLV": "Defensives", "XLP": "Defensives", "XLU": "Defensives", "XLRE": "Defensives",
    "EFA": "Developed International", "EWJ": "Developed International", "EWG": "Developed International",
    "EEM": "Emerging Markets", "INDA": "Emerging Markets", "EWZ": "Emerging Markets", "FXI": "Emerging Markets",
    "TLT": "Treasuries", "IEF": "Treasuries", "TIP": "Treasuries",
    "LQD": "Credit", "HYG": "Credit",
    "GLD": "Precious Metals", "SLV": "Precious Metals",
    "DBC": "Broad Commodities",
}
CASH_ETF = "BIL"        # where empty slots are parked
BENCHMARK = "SPY"       # relative-strength benchmark and buy-and-hold comparison
BOND_FOR_6040 = "IEF"   # bond leg of the 60/40 comparison

# --------------------------------------------------------------------------
# Market profiles: "us" = US-listed ETFs, "ireland" = Ireland-domiciled UCITS
# ETFs/ETCs listed on the London Stock Exchange (Yahoo suffix .L).
# All Ireland prices are converted to USD so returns are comparable.
# --------------------------------------------------------------------------
IRELAND_ETFS = {
    # ticker: (asset class, category, name)
    "CSPX.L": ("US Equity", "US Broad Market", "iShares Core S&P 500"),
    "CNDX.L": ("US Equity", "US Broad Market", "iShares Nasdaq 100"),
    "R2US.L": ("US Equity", "US Broad Market", "SPDR Russell 2000 US Small Cap"),
    "IWDA.L": ("Global Equity", "Global Developed", "iShares Core MSCI World"),
    "IUIT.L": ("Sector", "Technology", "iShares S&P 500 Info Tech"),
    "SMH.L": ("Sector", "Technology", "VanEck Semiconductor"),
    "IUCM.L": ("Sector", "Technology", "iShares S&P 500 Communication"),
    "IUES.L": ("Sector", "Energy", "iShares S&P 500 Energy"),
    "IUFS.L": ("Sector", "Cyclicals", "iShares S&P 500 Financials"),
    "IUIS.L": ("Sector", "Cyclicals", "iShares S&P 500 Industrials"),
    "IUMS.L": ("Sector", "Cyclicals", "iShares S&P 500 Materials"),
    "IUCD.L": ("Sector", "Cyclicals", "iShares S&P 500 Consumer Discretionary"),
    "IUHC.L": ("Sector", "Defensives", "iShares S&P 500 Health Care"),
    "IUCS.L": ("Sector", "Defensives", "iShares S&P 500 Consumer Staples"),
    "IUUS.L": ("Sector", "Defensives", "iShares S&P 500 Utilities"),
    "IUSP.L": ("Sector", "Defensives", "iShares US Property Yield"),
    "IMEU.L": ("International", "Developed International", "iShares Core MSCI Europe"),
    "IJPA.L": ("International", "Developed International", "iShares Core MSCI Japan IMI"),
    "CPXJ.L": ("International", "Developed International", "iShares Core MSCI Pacific ex-Japan"),
    "EIMI.L": ("International", "Emerging Markets", "iShares Core MSCI EM IMI"),
    "NDIA.L": ("International", "Emerging Markets", "iShares MSCI India"),
    "FXC.L": ("International", "Emerging Markets", "iShares China Large Cap"),
    "IBZL.L": ("International", "Emerging Markets", "iShares MSCI Brazil"),
    "IDTL.L": ("Bonds", "Treasuries", "iShares $ Treasury 20+yr"),
    "IBTM.L": ("Bonds", "Treasuries", "iShares $ Treasury 7-10yr"),
    "ITPS.L": ("Bonds", "Treasuries", "iShares $ TIPS"),
    "LQDE.L": ("Bonds", "Credit", "iShares $ Corp Bond"),
    "IHYU.L": ("Bonds", "Credit", "iShares $ High Yield Corp Bond"),
    "IGLN.L": ("Commodity", "Precious Metals", "iShares Physical Gold ETC"),
    "SSLN.L": ("Commodity", "Precious Metals", "iShares Physical Silver ETC"),
    "CMOD.L": ("Commodity", "Broad Commodities", "Invesco Bloomberg Commodity"),
}

INDIA_ETFS = {
    # NSE-listed ETFs (Yahoo suffix .NS), prices in INR. One category per sector/theme.
    "NIFTYBEES.NS": ("Indian Equity", "Large Cap", "Nippon Nifty 50 BeES"),
    "JUNIORBEES.NS": ("Indian Equity", "Large Cap", "Nippon Nifty Next 50 Junior BeES"),
    "MID150BEES.NS": ("Indian Equity", "Mid & Small Cap", "Nippon Nifty Midcap 150"),
    "HDFCSML250.NS": ("Indian Equity", "Mid & Small Cap", "HDFC Nifty Smallcap 250"),
    "BANKBEES.NS": ("Sector", "Banking & Financials", "Nippon Nifty Bank BeES"),
    "PSUBNKBEES.NS": ("Sector", "Banking & Financials", "Nippon Nifty PSU Bank BeES"),
    "ITBEES.NS": ("Sector", "IT", "Nippon Nifty IT"),
    "PHARMABEES.NS": ("Sector", "Pharma & Healthcare", "Nippon Nifty Pharma"),
    "FMCGIETF.NS": ("Sector", "FMCG & Consumption", "ICICI Pru Nifty FMCG"),
    "CONSUMBEES.NS": ("Sector", "FMCG & Consumption", "Nippon Nifty India Consumption"),
    "AUTOBEES.NS": ("Sector", "Auto", "Nippon Nifty Auto"),
    "INFRABEES.NS": ("Sector", "Infra & Realty", "Nippon Nifty Infrastructure"),
    "MOREALTY.NS": ("Sector", "Infra & Realty", "Motilal Oswal Nifty Realty"),
    "CPSEETF.NS": ("Sector", "PSU & Energy", "CPSE ETF"),
    "METALIETF.NS": ("Sector", "Metals", "ICICI Pru Nifty Metal"),
    "MODEFENCE.NS": ("Sector", "Defence", "Motilal Oswal Nifty India Defence"),
    "LOWVOLIETF.NS": ("Smart Beta", "Factor", "ICICI Pru Nifty 100 Low Vol 30"),
    "DIVOPPBEES.NS": ("Smart Beta", "Factor", "Nippon Nifty Dividend Opportunities 50"),
    "GOLDBEES.NS": ("Commodity", "Gold & Silver", "Nippon Gold BeES"),
    "SILVERBEES.NS": ("Commodity", "Gold & Silver", "Nippon Silver BeES"),
    "MON100.NS": ("International", "US Equity", "Motilal Oswal Nasdaq 100"),
    "MAFANG.NS": ("International", "US Equity", "Mirae NYSE FANG+"),
    "MASPTOP50.NS": ("International", "US Equity", "Mirae S&P 500 Top 50"),
    "HNGSNGBEES.NS": ("International", "China / Hong Kong", "Nippon Hang Seng BeES"),
    "LTGILTBEES.NS": ("Debt", "Gilts", "Nippon Nifty 8-13yr G-Sec"),
    "GILT5YBEES.NS": ("Debt", "Gilts", "Nippon Nifty 5yr Benchmark G-Sec"),
}

PROFILES = {
    "us": dict(
        label="US-listed ETFs (NYSE/Nasdaq)",
        universe=dict(UNIVERSE), category=dict(CATEGORY), names={},
        cash="BIL", cash_fallback=None, benchmark="SPY", bond="IEF",
        gold_silver=("GLD", "SLV"), to_usd=False),
    "ireland": dict(
        label="Ireland-domiciled UCITS ETFs (London Stock Exchange)",
        universe={t: v[0] for t, v in IRELAND_ETFS.items()},
        category={t: v[1] for t, v in IRELAND_ETFS.items()},
        names={t: v[2] for t, v in IRELAND_ETFS.items()},
        cash="IB01.L", cash_fallback="IBTS.L", benchmark="CSPX.L", bond="IBTM.L",
        gold_silver=("IGLN.L", "SSLN.L"), to_usd=True),
    "india": dict(
        label="Indian ETFs (NSE)",
        universe={t: v[0] for t, v in INDIA_ETFS.items()},
        category={t: v[1] for t, v in INDIA_ETFS.items()},
        names={t: v[2] for t, v in INDIA_ETFS.items()},
        cash="LIQUIDCASE.NS", cash_fallback=None, benchmark="NIFTYBEES.NS", bond="LTGILTBEES.NS",
        gold_silver=("GOLDBEES.NS", "SILVERBEES.NS"), to_usd=False, currency="₹",
        default_start="2015-01-01", bench_name="NIFTYBEES"),
}
for _p in PROFILES.values():
    _p.setdefault("currency", "$")
    _p.setdefault("default_start", "2007-01-01")
PROFILES["ireland"]["default_start"] = "2011-01-01"

# Indian INDEX strategy: broad-market, size and factor (smart-beta) index ETFs instead of sectors.
# Gold, silver, gilts and Nasdaq are included as optional diversifiers (exclude them in the app if you
# want a pure equity-index rotation). Benchmark = Nifty 500 index (^CRSLDX), so NIFTYBEES can be picked.
INDIA_INDEX_ETFS = {
    "NIFTYBEES.NS": ("Index", "Large Cap", "Nippon Nifty 50 BeES"),
    "NIF100BEES.NS": ("Index", "Large Cap", "Nippon Nifty 100 BeES"),
    "JUNIORBEES.NS": ("Index", "Next 50", "Nippon Nifty Next 50 Junior BeES"),
    "MONIFTY500.NS": ("Index", "Broad Market", "Motilal Oswal Nifty 500"),
    "MID150BEES.NS": ("Index", "Mid Cap", "Nippon Nifty Midcap 150"),
    "MOM100.NS": ("Index", "Mid Cap", "Motilal Oswal Nifty Midcap 100"),
    "MIDSELIETF.NS": ("Index", "Mid Cap", "ICICI Pru Nifty Midcap Select"),
    "HDFCSML250.NS": ("Index", "Small Cap", "HDFC Nifty Smallcap 250"),
    "MOM30IETF.NS": ("Factor", "Momentum", "ICICI Pru Nifty200 Momentum 30"),
    "ALPHA.NS": ("Factor", "Momentum", "Kotak Nifty Alpha 50"),
    "LOWVOLIETF.NS": ("Factor", "Low Volatility", "ICICI Pru Nifty 100 Low Vol 30"),
    "ALPL30IETF.NS": ("Factor", "Low Volatility", "ICICI Pru Nifty Alpha Low-Vol 30"),
    "QUAL30IETF.NS": ("Factor", "Quality / Value / Dividend", "ICICI Pru Nifty200 Quality 30"),
    "NV20IETF.NS": ("Factor", "Quality / Value / Dividend", "ICICI Pru Nifty50 Value 20"),
    "DIVOPPBEES.NS": ("Factor", "Quality / Value / Dividend", "Nippon Nifty Dividend Opportunities 50"),
    "GOLDBEES.NS": ("Diversifier", "Gold & Silver", "Nippon Gold BeES"),
    "SILVERBEES.NS": ("Diversifier", "Gold & Silver", "Nippon Silver BeES"),
    "LTGILTBEES.NS": ("Diversifier", "Gilts", "Nippon Nifty 8-13yr G-Sec"),
    "MON100.NS": ("Diversifier", "US Equity", "Motilal Oswal Nasdaq 100"),
}
PROFILES["india_index"] = dict(
    label="Indian index ETFs (NSE) - broad & factor indices",
    universe={t: v[0] for t, v in INDIA_INDEX_ETFS.items()},
    category={t: v[1] for t, v in INDIA_INDEX_ETFS.items()},
    names={t: v[2] for t, v in INDIA_INDEX_ETFS.items()},
    cash="LIQUIDCASE.NS", cash_fallback=None, benchmark="^CRSLDX", bond="LTGILTBEES.NS",
    gold_silver=("GOLDBEES.NS", "SILVERBEES.NS"), to_usd=False, currency="₹",
    default_start="2015-01-01", bench_name="NIFTY 500")
CCY = "$"
PROFILE = "us"
NAMES = {}
CASH_FALLBACK = None
GOLD_SILVER = ("GLD", "SLV")
TO_USD = False


def set_profile(name):
    """Switch the module to a market profile (universe, cash ETF, benchmark...)."""
    global PROFILE, UNIVERSE, CATEGORY, NAMES, CASH_ETF, CASH_FALLBACK, BENCHMARK
    global BOND_FOR_6040, GOLD_SILVER, TO_USD, CCY
    p = PROFILES[name]
    PROFILE = name
    UNIVERSE, CATEGORY, NAMES = dict(p["universe"]), dict(p["category"]), dict(p["names"])
    CASH_ETF, CASH_FALLBACK, BENCHMARK = p["cash"], p["cash_fallback"], p["benchmark"]
    BOND_FOR_6040, GOLD_SILVER, TO_USD = p["bond"], p["gold_silver"], p["to_usd"]
    CCY = p["currency"]
    _apply_custom()


def bench_name():
    return PROFILES[PROFILE].get("bench_name", BENCHMARK)


def bench_label():
    return f"{bench_name()} buy & hold"


def cash_series(close):
    """Cash-ETF prices, spliced with a fallback ETF's returns before the cash ETF existed."""
    if CASH_ETF not in close.columns:
        return None
    main = close[CASH_ETF]
    if CASH_FALLBACK and CASH_FALLBACK in close.columns:
        r = main.pct_change(fill_method=None)
        r = r.where(main.notna(), close[CASH_FALLBACK].pct_change(fill_method=None))
        return (1 + r.fillna(0)).cumprod()
    return main


@dataclass
class Config:
    start: str = "2007-01-01"          # first trading date of the backtest
    end: str | None = None             # None = latest data
    capital: float = 100_000.0
    n_holdings: int = 5
    exit_rank: int = 10
    rank_criteria: str = "sharpe"      # "sharpe" or "return"
    periods: tuple = (252, 90, 60)
    weights: tuple = (2, 1, 1)
    retracement: float = 0.20          # must be within 20% of 252-day high (None = off)
    use_rs: bool = True                # relative strength filter vs SPY
    rs_rule: str = "Ratio above 200 EMA"
    ema_len: int = 200                 # kept for compatibility (not used)
    entry_condition: str = "Close above 200 EMA"
    exit_condition: str = "Close below 200 EMA"
    stop_loss: float | None = 0.12     # None = off
    rebalance: str = "weekly"          # weekly | biweekly | monthly
    weekday: int = 2                   # 0=Mon ... 2=Wed
    cost_bps: float = 5.0              # commission + slippage per side, basis points
    use_cash_etf: bool = True
    exclude: tuple = ()
    include: tuple = ()                # if set, only these ETFs are considered (manual selection)
    max_per_category: int = 0          # ETF Group: 0 = off, 1 = only the best ETF per category
    category_buffer: int = 1           # keep a held ETF while it ranks within top (max + buffer) of its category
    categories: tuple = ()             # optional overrides: ((ticker, category), ...)
    name: str = "base"


def category_map(cfg):
    m = dict(CATEGORY)
    m.update(dict(cfg.categories))
    return m


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
DATA_DIR = "data"


def _fetch(tickers, yf, convert=None, **window):
    """Download adjusted closes for tickers (retrying failures one by one).
    window: yfinance range, default start="2005-01-01"; e.g. period="5d" for a quick update."""
    window = window or {"start": "2005-01-01"}
    raw = yf.download(sorted(tickers), auto_adjust=True, progress=False, threads=False, **window)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": sorted(tickers)[0]})
    close = close.dropna(how="all").sort_index()
    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    for t in list(missing):                      # retry failed tickers one at a time
        try:
            one = yf.download(t, auto_adjust=True, progress=False, threads=False, **window)
            col = one["Close"]
            col = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
            if not col.dropna().empty:
                close = close.reindex(close.index.union(col.index))
                close[t] = col
                missing.remove(t)
                print(f"Retried {t}: OK")
        except Exception as e:
            print(f"Retry failed for {t}: {e}")
    if missing:
        print(f"WARNING: no data for {missing}")
    close = close.drop(columns=[t for t in missing if t in close.columns])
    if (TO_USD if convert is None else convert) and not close.empty:
        close = convert_to_usd(close, yf)
    return close, missing


# --------------------------------------------------------------------------
# Global markets catalog (Yahoo symbols) for the cross-market comparison view.
# Indices are price-only (no dividends). Yields are plotted as % change of the yield.
# --------------------------------------------------------------------------
GLOBAL_CATALOG = {
    "Country indices": {
        "^NSEI": "NIFTY 50 (India)", "^BSESN": "SENSEX (India)", "^NSEBANK": "BANKNIFTY (India)",
        "^CRSLDX": "NIFTY 500 (India)", "^GSPC": "S&P 500 (US)", "^DJI": "Dow Jones (US)",
        "^IXIC": "Nasdaq Composite (US)", "^RUT": "Russell 2000 (US)", "^GSPTSE": "TSX (Canada)",
        "^BVSP": "Bovespa (Brazil)", "^FTSE": "FTSE 100 (UK)", "^GDAXI": "DAX (Germany)",
        "^FCHI": "CAC 40 (France)", "^STOXX50E": "Euro Stoxx 50", "^N225": "Nikkei 225 (Japan)",
        "^HSI": "Hang Seng (Hong Kong)", "000001.SS": "Shanghai Composite (China)",
        "^KS11": "KOSPI (Korea)", "^TWII": "Taiwan Weighted", "^AXJO": "ASX 200 (Australia)",
        "^STI": "Straits Times (Singapore)", "^TASI.SR": "Tadawul (Saudi Arabia)",
    },
    "India sector indices": {
        "^CNXIT": "Nifty IT", "^CNXPHARMA": "Nifty Pharma", "^CNXAUTO": "Nifty Auto",
        "^CNXFMCG": "Nifty FMCG", "^CNXMETAL": "Nifty Metal", "^CNXREALTY": "Nifty Realty",
        "^CNXENERGY": "Nifty Energy", "^CNXPSUBANK": "Nifty PSU Bank", "^CNXINFRA": "Nifty Infra",
        "^CNXMEDIA": "Nifty Media", "NIFTY_FIN_SERVICE.NS": "Nifty Financial Services",
        "^NSEMDCP50": "Nifty Midcap 50", "^CNXPSE": "Nifty PSE", "^CNXCONSUM": "Nifty Consumption",
    },
    "US sectors (SPDR ETFs)": {
        "XLK": "US Technology", "XLF": "US Financials", "XLE": "US Energy", "XLV": "US Health Care",
        "XLI": "US Industrials", "XLY": "US Cons. Discretionary", "XLP": "US Cons. Staples",
        "XLU": "US Utilities", "XLB": "US Materials", "XLRE": "US Real Estate",
        "XLC": "US Communication", "SMH": "US Semiconductors",
    },
    "Bond yields": {
        "^IRX": "US 3M yield", "^FVX": "US 5Y yield", "^TNX": "US 10Y yield", "^TYX": "US 30Y yield",
    },
    "Currencies": {
        "DX-Y.NYB": "DXY (US Dollar Index)", "INR=X": "USDINR", "JPY=X": "USDJPY",
        "EURUSD=X": "EURUSD", "GBPUSD=X": "GBPUSD", "CNY=X": "USDCNY", "AED=X": "USDAED",
    },
    "Commodities & crypto": {
        "GC=F": "Gold", "SI=F": "Silver", "CL=F": "WTI crude", "BZ=F": "Brent crude",
        "HG=F": "Copper", "NG=F": "Natural gas", "BTC-USD": "Bitcoin",
    },
}
GLOBAL_SHORT = {
    "^NSEI": "NIFTY", "^BSESN": "SENSEX", "^NSEBANK": "BANKNIFTY", "^CRSLDX": "NIFTY500", "^GSPC": "SPX",
    "^DJI": "DJI", "^IXIC": "IXIC", "^RUT": "RUT", "^GSPTSE": "TSX", "^BVSP": "BOVESPA", "^FTSE": "FTSE",
    "^GDAXI": "DAX", "^FCHI": "CAC", "^STOXX50E": "SX5E", "^N225": "NIKKEI", "^HSI": "HSI",
    "000001.SS": "SHCOMP", "^KS11": "KOSPI", "^TWII": "TAIEX", "^AXJO": "ASX200", "^STI": "STI",
    "^TASI.SR": "TASI", "^IRX": "US03M", "^FVX": "US05Y", "^TNX": "US10Y", "^TYX": "US30Y",
    "DX-Y.NYB": "DXY", "INR=X": "USDINR", "JPY=X": "USDJPY", "EURUSD=X": "EURUSD", "GBPUSD=X": "GBPUSD",
    "CNY=X": "USDCNY", "AED=X": "USDAED", "GC=F": "GOLD", "SI=F": "SILVER", "CL=F": "WTI", "BZ=F": "BRENT",
    "HG=F": "COPPER", "NG=F": "NATGAS", "BTC-USD": "BTC", "NIFTY_FIN_SERVICE.NS": "FINNIFTY",
}
GLOBAL_PRESETS = {
    "Macro (like TradingView)": ["^TNX", "^TYX", "^FVX", "^NSEBANK", "DX-Y.NYB", "^KS11", "^GDAXI", "JPY=X",
                                 "^NSEI", "BZ=F", "INR=X", "CL=F", "^DJI", "^GSPC", "^HSI", "SI=F", "GC=F"],
    "Major country indices": ["^NSEI", "^GSPC", "^IXIC", "^FTSE", "^GDAXI", "^N225", "^HSI", "000001.SS",
                              "^KS11", "^BVSP", "^AXJO", "^TASI.SR"],
    "India sectors": ["^NSEI", "^CNXIT", "^CNXPHARMA", "^CNXAUTO", "^CNXFMCG", "^CNXMETAL", "^CNXREALTY",
                      "^CNXENERGY", "^CNXPSUBANK", "^CNXINFRA", "NIFTY_FIN_SERVICE.NS"],
    "US sectors": ["^GSPC", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH"],
    "Commodities, FX & yields": ["GC=F", "SI=F", "CL=F", "HG=F", "DX-Y.NYB", "INR=X", "^TNX", "BTC-USD"],
}


def merge_latest(old, new):
    """Overwrite/append the newest bars onto cached history."""
    if old is None or old.empty:
        return new.sort_index()
    out = old.reindex(old.index.union(new.index))
    for c in new.columns:
        if c not in out.columns:
            out[c] = np.nan
    out.update(new)
    return out.sort_index()


def quick_update_prices():
    """Fetch only the last few days (incl. today's live bar, ~15 min delayed on Yahoo) for the
    current market and merge them into the saved history. Much faster than a full download."""
    import yfinance as yf
    cache = os.path.join(DATA_DIR, "prices.csv" if PROFILE == "us" else f"prices_{PROFILE}.csv")
    if not os.path.exists(cache):
        return load_prices(refresh=True)
    old = pd.read_csv(cache, index_col=0, parse_dates=True).sort_index()
    base = [c for c in old.columns]
    new, _ = _fetch(base, yf, period="5d")
    if not new.empty:
        merge_latest(old, new).to_csv(cache)
    meta = custom_tickers()
    cp = _custom_prices_path()
    if meta and os.path.exists(cp):
        oldc = pd.read_csv(cp, index_col=0, parse_dates=True).sort_index()
        newc, _ = _fetch(list(meta), yf, period="5d")
        if not newc.empty:
            merge_latest(oldc, newc).to_csv(cp)
    return load_prices()


def quick_update_global(symbols):
    """Fast update of the last few days for global symbols (no currency conversion)."""
    import yfinance as yf
    path = os.path.join(DATA_DIR, "global_prices.csv")
    old = pd.read_csv(path, index_col=0, parse_dates=True).sort_index() if os.path.exists(path) else pd.DataFrame()
    missing = [x for x in symbols if x not in old.columns]
    if missing:
        load_global(missing)
        old = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    new, _ = _fetch(list(symbols), yf, convert=False, period="5d")
    if not new.empty:
        new.index = pd.to_datetime(new.index).tz_localize(None) if getattr(new.index, "tz", None) else new.index
        os.makedirs(DATA_DIR, exist_ok=True)
        merge_latest(old, new).to_csv(path)


# --------------------------------------------------------------------------
# India ETF explorer: the full NSE ETF list (+ REITs / InvITs) with categories
# --------------------------------------------------------------------------
NSE_ETF_LIST_URLS = [
    "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv",
    "https://archives.nseindia.com/content/equities/eq_etfseclist.csv",
]
REITS_INVITS = {
    "EMBASSY": "Embassy Office Parks REIT", "MINDSPACE": "Mindspace Business Parks REIT",
    "BIRET": "Brookfield India Real Estate Trust", "NXST": "Nexus Select Trust (REIT)",
    "PGINVIT": "POWERGRID Infrastructure InvIT", "INDIGRID": "IndiGrid InvIT",
    "IRBINVIT": "IRB InvIT Fund", "CUBEINVIT": "Cube Highways InvIT",
}
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept": "text/csv,application/json,text/plain,*/*", "Accept-Language": "en-US,en;q=0.9"}

# keyword -> category, checked in order against "underlying + name" (lower case)
INDIA_CATEGORY_RULES = [
    (("liquid", "overnight", "money market", "1d rate", "tri-party", "treps"), "Liquid / Overnight"),
    (("bharat bond", "gilt", "g-sec", "gsec", "sdl", "treasury", "bond", "debt", "psu bond",
      "corporate", "crisil", "t-bill", "tbill", "gilts"), "Debt / Gilt"),
    (("silver",), "Silver"),
    (("gold",), "Gold"),
    (("nasdaq", "s&p 500", "s&p500", "fang", "hang seng", "nyse", "us ", "u.s.", "global", "msci",
      "japan", "china", "taiwan", "international", "world"), "International"),
    (("psu bank",), "PSU Bank"),
    (("private bank", "pvt bank"), "Private Bank"),
    (("bank",), "Banking"),
    (("financial", "fin serv", "capital market", "insurance"), "Financial Services"),
    (("information technology", " it ", "nifty it", "technology", "digital", "tech"), "IT & Tech"),
    (("pharma", "healthcare", "health"), "Pharma & Healthcare"),
    (("fmcg", "consumption", "consumer"), "FMCG & Consumption"),
    (("auto", "ev ", "electric vehicle", "mobility"), "Auto & EV"),
    (("realty", "housing"), "Realty"),
    (("infra",), "Infrastructure"),
    (("cpse", "psu", "pse", "public sector"), "PSU / CPSE"),
    (("energy", "oil", "gas", "power", "commodit"), "Energy & Commodities"),
    (("metal",), "Metals"),
    (("defence", "defense"), "Defence"),
    (("manufactur", "industr", "capital goods"), "Manufacturing & Industrials"),
    (("momentum", "alpha", "low vol", "quality", "value", "equal weight", "dividend", "beta",
      "factor", "multi", "growth"), "Factor / Smart Beta"),
    (("smallcap", "small cap", "small-cap", "microcap", "micro cap"), "Small Cap"),
    (("midcap", "mid cap", "mid-cap", "mid150", "mid 150", "midsmall"), "Mid Cap"),
    (("next 50", "nifty 100", "nifty100", "junior", "next50", "200", "500", "total market",
      "largemidcap", "large mid"), "Large & Broad Market"),
    (("nifty 50", "nifty50", "sensex", "bse 100", "nifty"), "Large Cap (Nifty / Sensex)"),
    (("esg", "shariah", "mnc", "tourism", "rural", "india"), "Thematic"),
]


def classify_india_etf(name, underlying=""):
    txt = f" {str(underlying)} {str(name)} ".lower()
    for keys, cat in INDIA_CATEGORY_RULES:
        if any(k in txt for k in keys):
            return cat
    return "Other Equity"


def fetch_nse_etf_list(refresh=False, data_dir=None):
    """Full list of NSE-listed ETFs: Symbol, Name, Underlying, Category. Cached to data/nse_etf_list.csv.
    Tries NSE's ETF file, then NSE's ETF API; returns the cache (or an empty frame) if both fail."""
    import io
    import urllib.request
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, "nse_etf_list.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    df, err = None, []
    for url in NSE_ETF_LIST_URLS:
        try:
            req = urllib.request.Request(url, headers=_UA)
            raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
            d = pd.read_csv(io.StringIO(raw))
            d.columns = [c.strip().lower().replace(" ", "") for c in d.columns]
            sym = next(c for c in d.columns if c.startswith("symbol"))
            nmc = next((c for c in d.columns if "securityname" in c or c in ("name", "companyname")), sym)
            und = next((c for c in d.columns if "underlying" in c), nmc)
            df = pd.DataFrame({"Symbol": d[sym].astype(str).str.strip(), "Name": d[nmc].astype(str).str.strip(),
                               "Underlying": d[und].astype(str).str.strip()})
            break
        except Exception as e:
            err.append(f"{url}: {e}")
    if df is None:                                  # NSE live API (needs a cookie from the home page)
        try:
            import http.cookiejar
            import json
            cj = http.cookiejar.CookieJar()
            op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
            op.addheaders = list(_UA.items())
            op.open("https://www.nseindia.com/", timeout=30).read()
            js = json.loads(op.open("https://www.nseindia.com/api/etf", timeout=30).read().decode())
            rows = js.get("data", [])
            df = pd.DataFrame({"Symbol": [r.get("symbol", "") for r in rows],
                               "Name": [(r.get("meta") or {}).get("companyName", r.get("symbol", "")) for r in rows],
                               "Underlying": [r.get("assets", "") for r in rows]})
        except Exception as e:
            err.append(f"NSE API: {e}")
    if df is None or df.empty:
        if os.path.exists(path):
            return pd.read_csv(path)
        raise RuntimeError("Could not download the NSE ETF list. " + " | ".join(err))
    df = df[df.Symbol.str.len() > 0].drop_duplicates("Symbol")
    df["Category"] = [classify_india_etf(n, u) for n, u in zip(df.Name, df.Underlying)]
    reit = pd.DataFrame({"Symbol": list(REITS_INVITS), "Name": list(REITS_INVITS.values()),
                         "Underlying": "REIT / InvIT", "Category": "REIT / InvIT"})
    df = pd.concat([df, reit[~reit.Symbol.isin(df.Symbol)]], ignore_index=True)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def save_etf_list_upload(df_up, data_dir=None):
    """Use a list you downloaded yourself (CSV with a Symbol column, optional name/underlying)."""
    data_dir = data_dir or DATA_DIR
    d = df_up.copy()
    d.columns = [c.strip().lower().replace(" ", "") for c in d.columns]
    sym = next(c for c in d.columns if c.startswith("symbol"))
    nmc = next((c for c in d.columns if "name" in c), sym)
    und = next((c for c in d.columns if "underlying" in c or "asset" in c), nmc)
    df = pd.DataFrame({"Symbol": d[sym].astype(str).str.strip().str.replace(".NS", "", regex=False),
                       "Name": d[nmc].astype(str), "Underlying": d[und].astype(str)})
    df["Category"] = [classify_india_etf(n, u) for n, u in zip(df.Name, df.Underlying)]
    reit = pd.DataFrame({"Symbol": list(REITS_INVITS), "Name": list(REITS_INVITS.values()),
                         "Underlying": "REIT / InvIT", "Category": "REIT / InvIT"})
    df = pd.concat([df, reit[~reit.Symbol.isin(df.Symbol)]], ignore_index=True).drop_duplicates("Symbol")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "nse_etf_list.csv"), index=False)
    return df


def load_india_all(symbols, refresh=False, quick=False, data_dir=None):
    """Close and volume for many NSE symbols (Yahoo .NS), last ~4 years.
    refresh = full re-download; quick = only the last 5 days merged in."""
    data_dir = data_dir or DATA_DIR
    pc, pv = os.path.join(data_dir, "india_all_close.csv"), os.path.join(data_dir, "india_all_volume.csv")
    if os.environ.get("INDIA_ALL_CLOSE_CSV"):
        return (pd.read_csv(os.environ["INDIA_ALL_CLOSE_CSV"], index_col=0, parse_dates=True),
                pd.read_csv(os.environ["INDIA_ALL_VOLUME_CSV"], index_col=0, parse_dates=True))
    close = pd.read_csv(pc, index_col=0, parse_dates=True) if os.path.exists(pc) and not refresh else pd.DataFrame()
    vol = pd.read_csv(pv, index_col=0, parse_dates=True) if os.path.exists(pv) and not refresh else pd.DataFrame()
    tick = [f"{x}.NS" for x in symbols]
    need = [t for t in tick if t not in close.columns]
    import yfinance as yf

    def grab(tlist, **win):
        frames_c, frames_v = [], []
        for i in range(0, len(tlist), 60):              # batches keep Yahoo happy
            raw = yf.download(tlist[i:i + 60], auto_adjust=True, progress=False, threads=True, **win)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                frames_c.append(raw["Close"])
                frames_v.append(raw["Volume"])
            else:
                frames_c.append(raw[["Close"]].rename(columns={"Close": tlist[i]}))
                frames_v.append(raw[["Volume"]].rename(columns={"Volume": tlist[i]}))
        if not frames_c:
            return pd.DataFrame(), pd.DataFrame()
        c_ = pd.concat(frames_c, axis=1)
        v_ = pd.concat(frames_v, axis=1)
        c_ = c_.loc[:, ~c_.columns.duplicated()].dropna(axis=1, how="all")
        v_ = v_.loc[:, ~v_.columns.duplicated()].reindex(columns=c_.columns)
        return c_, v_

    changed = False
    if need:
        start = (pd.Timestamp.today() - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
        c_, v_ = grab(need, start=start)
        if not c_.empty:
            close, vol = merge_latest(close, c_), merge_latest(vol, v_)
            changed = True
    if quick and not close.empty:
        c_, v_ = grab([t for t in tick if t in close.columns], period="5d")
        if not c_.empty:
            close, vol = merge_latest(close, c_), merge_latest(vol, v_)
            changed = True
    if changed:
        os.makedirs(data_dir, exist_ok=True)
        close.sort_index().to_csv(pc)
        vol.sort_index().to_csv(pv)
    return close, vol


def india_etf_table(lst, close, vol, cfg):
    """One row per ETF: returns over 1W/1M/6M/1Y/3Y, momentum score, overall and category rank, liquidity."""
    hz = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756}
    rows = []
    score = compute_score(close, cfg).ffill().iloc[-1] if not close.empty else pd.Series(dtype=float)
    for r in lst.itertuples():
        t = f"{r.Symbol}.NS"
        row = {"Symbol": r.Symbol, "ETF name": r.Name, "Underlying": r.Underlying, "Category": r.Category}
        if t in close.columns:
            sr = close[t].dropna()
            for k, n in hz.items():
                row[k] = (sr.iloc[-1] / sr.iloc[-1 - n] - 1) * 100 if len(sr) > n else np.nan
            row["Price"] = sr.iloc[-1] if len(sr) else np.nan
            row["Last date"] = sr.index[-1].date() if len(sr) else None
            v = vol[t].dropna() if t in vol.columns else pd.Series(dtype=float)
            px_ = close[t].reindex(v.index)
            row["Avg value 1M (₹ Cr)"] = float((v * px_).tail(21).mean() / 1e7) if len(v) else np.nan
            row["Score"] = score.get(t, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def global_name(sym):
    for grp in GLOBAL_CATALOG.values():
        if sym in grp:
            return grp[sym]
    return sym


def global_short(sym):
    return GLOBAL_SHORT.get(sym, sym.replace("^", "").split(".")[0].split("=")[0])


def load_global(symbols, refresh=False, data_dir=None):
    """Daily closes for global symbols (no currency conversion). Cached to data/global_prices.csv;
    only symbols not in the cache are downloaded unless refresh=True."""
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, "global_prices.csv")
    csv = os.environ.get("GLOBAL_PRICES_CSV")
    if csv:
        return pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()
    have = pd.read_csv(path, index_col=0, parse_dates=True).sort_index() if os.path.exists(path) else pd.DataFrame()
    need = sorted(set(symbols) if refresh else set(symbols) - set(have.columns))
    if need:
        import yfinance as yf
        print(f"Downloading {len(need)} global symbols ...")
        new, _ = _fetch(need, yf, convert=False)
        if not new.empty:
            new.index = pd.to_datetime(new.index).tz_localize(None) if getattr(new.index, "tz", None) else new.index
            have = have.drop(columns=[c for c in new.columns if c in have.columns]) if not have.empty else have
            have = new if have.empty else have.join(new, how="outer")
            os.makedirs(data_dir, exist_ok=True)
            have.sort_index().to_csv(path)
    return have


def _custom_meta_path():
    return os.path.join(DATA_DIR, f"custom_{PROFILE}.json")


def _custom_prices_path():
    return os.path.join(DATA_DIR, f"custom_prices_{PROFILE}.csv")


def custom_tickers():
    """Your own added tickers for the current market: {ticker: [asset class, category, name]}."""
    import json
    try:
        with open(_custom_meta_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _apply_custom():
    for t, (asset, cat, name) in custom_tickers().items():
        UNIVERSE[t], CATEGORY[t], NAMES[t] = asset, cat, name


def add_custom_tickers(entries):
    """Download and register extra tickers. entries = {ticker: (asset, category, name)}.
    Returns (added, failed)."""
    import json
    import yfinance as yf
    tick = [t.strip().upper() for t in entries]
    entries = {t.strip().upper(): v for t, v in entries.items()}
    new, failed = _fetch(tick, yf)
    added = [t for t in tick if t in new.columns]
    if added:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = _custom_prices_path()
        old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
        old = old.drop(columns=[c for c in added if c in old.columns])
        merged = new[added] if old.empty else old.join(new[added], how="outer")
        merged.to_csv(path)
        meta = custom_tickers()
        for t in added:
            meta[t] = list(entries[t])
        with open(_custom_meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        _apply_custom()
    return added, failed


def remove_custom_ticker(t):
    import json
    meta = custom_tickers()
    meta.pop(t, None)
    with open(_custom_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    for d in (UNIVERSE, CATEGORY, NAMES):
        d.pop(t, None)


def load_prices(data_dir=None, refresh=False, csv_path=None):
    """Adjusted daily closes, one column per ticker (base universe + your added tickers)."""
    if csv_path:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return df.sort_index()
    data_dir = data_dir or DATA_DIR
    cache = os.path.join(data_dir, "prices.csv" if PROFILE == "us" else f"prices_{PROFILE}.csv")
    if os.path.exists(cache) and not refresh:
        print(f"Loading cached prices from {cache} (use --refresh to update)")
        close = pd.read_csv(cache, index_col=0, parse_dates=True).sort_index()
    else:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("yfinance is not installed. Run: pip install -r requirements.txt")
        base = set(PROFILES[PROFILE]["universe"]) | {CASH_ETF, BENCHMARK, BOND_FOR_6040}
        if CASH_FALLBACK:
            base.add(CASH_FALLBACK)
        print(f"Downloading {len(base)} tickers from Yahoo Finance ...")
        close, _ = _fetch(base, yf)
        if close.empty:
            sys.exit("Download failed - check your internet connection and try again.")
        os.makedirs(data_dir, exist_ok=True)
        close.to_csv(cache)
        print(f"Saved {close.shape[0]} days x {close.shape[1]} tickers to {cache}")
        meta = custom_tickers()
        if meta:                                     # refresh your added tickers too
            add_custom_tickers({t: tuple(v) for t, v in meta.items()})
    cp = _custom_prices_path()
    if os.path.exists(cp):
        extra = pd.read_csv(cp, index_col=0, parse_dates=True).sort_index()
        extra = extra[[c for c in extra.columns if c not in close.columns]]
        if not extra.empty:
            close = close.join(extra, how="left")
    return close


def clean_unit_glitches(s):
    """Fix Yahoo's occasional pence/pound (x100) unit flips on LSE prices."""
    s = s.copy()
    med = s.rolling(15, min_periods=5, center=True).median()
    ratio = s / med
    s[ratio > 30] = s[ratio > 30] / 100
    s[ratio < 1 / 30] = s[ratio < 1 / 30] * 100
    return s


def convert_to_usd(close, yf):
    """Convert each London line to USD using its quote currency (USD, GBP, GBp, EUR)."""
    fx_needed, cur = {}, {}
    for t in close.columns:
        try:
            c = yf.Ticker(t).fast_info["currency"]
        except Exception:
            c = None
        cur[t] = c or "USD"
    print("Quote currencies:", {k: v for k, v in cur.items() if v != "USD"} or "all USD")
    pairs = {"GBP": "GBPUSD=X", "GBp": "GBPUSD=X", "GBX": "GBPUSD=X", "EUR": "EURUSD=X", "CHF": "CHFUSD=X"}
    need = sorted({pairs[c] for c in cur.values() if c in pairs})
    if need:
        fx = yf.download(need, start="2005-01-01", auto_adjust=True, progress=False, threads=False)["Close"]
        if isinstance(fx, pd.Series):
            fx = fx.to_frame(need[0])
        fx = fx.reindex(close.index).ffill()
    out = close.copy()
    for t, c in cur.items():
        s = clean_unit_glitches(close[t])
        if c in ("GBp", "GBX"):
            s = s / 100
        if c in pairs:
            s = s * fx[pairs[c]]
        elif c != "USD":
            print(f"WARNING: unknown currency {c} for {t}; left as is")
        out[t] = s
    return out


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
class Indicators:
    """Lazy, cached technical indicators on a price DataFrame (one column per ETF)."""

    def __init__(self, px, cash=None):
        self.px, self.cash, self._c = px, cash, {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def ema(self, n):
        return self._get(("ema", n), lambda: self.px.ewm(span=n, adjust=False, min_periods=n).mean())

    def sma(self, n):
        return self._get(("sma", n), lambda: self.px.rolling(n, min_periods=n).mean())

    def rsi(self, n):
        def f():
            d = self.px.diff()
            up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
            dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
            return 100 - 100 / (1 + up / dn)
        return self._get(("rsi", n), f)

    def macd(self):
        def f():
            line = (self.px.ewm(span=12, adjust=False, min_periods=26).mean()
                    - self.px.ewm(span=26, adjust=False, min_periods=26).mean())
            return line, line.ewm(span=9, adjust=False, min_periods=9).mean()
        return self._get("macd", f)

    def ret(self, n):
        return self._get(("ret", n), lambda: self.px / self.px.shift(n) - 1)

    def high(self, n):
        return self._get(("high", n), lambda: self.px.rolling(n, min_periods=n).max())

    def cash_ret(self, n):
        if self.cash is None:
            return 0.0
        r = self.cash / self.cash.shift(n) - 1
        return pd.DataFrame({c: r for c in self.px.columns}, index=self.px.index)


# Entry conditions: True = the ETF may be bought
ENTRY_CONDITIONS = {
    "None": lambda i: i.px.notna(),
    "Close above 200 EMA": lambda i: i.px > i.ema(200),
    "Close above 100 EMA": lambda i: i.px > i.ema(100),
    "Close above 50 EMA": lambda i: i.px > i.ema(50),
    "Close above 200 SMA": lambda i: i.px > i.sma(200),
    "Golden Cross state (50 EMA > 200 EMA)": lambda i: i.ema(50) > i.ema(200),
    "Trend stack (Close > 50 EMA > 200 EMA)": lambda i: (i.px > i.ema(50)) & (i.ema(50) > i.ema(200)),
    "20 EMA above 50 EMA": lambda i: i.ema(20) > i.ema(50),
    "252 RSI > 50": lambda i: i.rsi(252) > 50,
    "14 RSI > 50": lambda i: i.rsi(14) > 50,
    "14 RSI between 50 and 70 (not overbought)": lambda i: (i.rsi(14) > 50) & (i.rsi(14) < 70),
    "MACD above signal line": lambda i: i.macd()[0] > i.macd()[1],
    "MACD above zero": lambda i: i.macd()[0] > 0,
    "Close above 20 SMA (Bollinger mid)": lambda i: i.px > i.sma(20),
    "Within 5% of 52-week high (breakout zone)": lambda i: i.px >= 0.95 * i.high(252),
    "Dual Momentum (12M return > T-bills)": lambda i: i.ret(252) > i.cash_ret(252),
    "3M return positive (absolute momentum)": lambda i: i.ret(63) > 0,
}

# Exit conditions: True = sell the holding at the next rebalance
EXIT_CONDITIONS = {
    "None (rank exit only)": lambda i: i.px.isna(),
    "Close below 200 EMA": lambda i: i.px < i.ema(200),
    "Close below 100 EMA": lambda i: i.px < i.ema(100),
    "Close below 50 EMA": lambda i: i.px < i.ema(50),
    "Close below 200 SMA": lambda i: i.px < i.sma(200),
    "Death Cross state (50 EMA < 200 EMA)": lambda i: i.ema(50) < i.ema(200),
    "20 EMA below 50 EMA": lambda i: i.ema(20) < i.ema(50),
    "14 RSI below 30": lambda i: i.rsi(14) < 30,
    "14 RSI below 40": lambda i: i.rsi(14) < 40,
    "MACD below signal line": lambda i: i.macd()[0] < i.macd()[1],
    "MACD below zero": lambda i: i.macd()[0] < 0,
    "Close below 20 SMA": lambda i: i.px < i.sma(20),
    "Close 10% below 22-day high (chandelier)": lambda i: i.px < 0.90 * i.high(22),
    "Dual Momentum off (12M return < T-bills)": lambda i: i.ret(252) < i.cash_ret(252),
}

# Relative strength rules: ratio = ETF / SPY
RS_RULES = {
    "Ratio above 200 EMA": lambda r: r > r.ewm(span=200, adjust=False, min_periods=200).mean(),
    "Ratio above 100 EMA": lambda r: r > r.ewm(span=100, adjust=False, min_periods=100).mean(),
    "Ratio above 50 EMA": lambda r: r > r.ewm(span=50, adjust=False, min_periods=50).mean(),
    "Ratio rising over 3 months": lambda r: r > r.shift(63),
    "Ratio rising over 6 months": lambda r: r > r.shift(126),
}


def compute_score(px, cfg):
    """Blended momentum score (Sharpe or return over cfg.periods, weighted)."""
    rets = px.pct_change(fill_method=None)
    score = None
    for p, w in zip(cfg.periods, cfg.weights):
        if cfg.rank_criteria == "sharpe":
            s = rets.rolling(p, min_periods=p).mean() / rets.rolling(p, min_periods=p).std()
            s = s * math.sqrt(252)
        else:  # plain return over the period
            s = px / px.shift(p) - 1
        score = s * w if score is None else score + s * w
    return score / sum(cfg.weights)


def compute_signals(close, cfg):
    """Returns prices, score, rank, entry-eligible flags and exit flags (True = exit)."""
    tickers = [t for t in UNIVERSE if t in close.columns and t not in cfg.exclude
               and (not cfg.include or t in cfg.include)]
    if not tickers:
        raise ValueError("No ETFs selected - pick at least one ETF.")
    px = close[tickers]
    rets = px.pct_change(fill_method=None)

    score = compute_score(px, cfg)

    ind = Indicators(px, cash_series(close))
    eligible = score.notna() & ENTRY_CONDITIONS[cfg.entry_condition](ind).fillna(False).astype(bool)
    if cfg.retracement is not None:
        eligible &= px >= (1 - cfg.retracement) * ind.high(252)
    if cfg.use_rs and BENCHMARK in close.columns:
        ratio = px.div(close[BENCHMARK], axis=0)
        eligible &= RS_RULES[cfg.rs_rule](ratio).fillna(False).astype(bool)
    exit_flag = EXIT_CONDITIONS[cfg.exit_condition](ind).fillna(False).astype(bool)
    rank = score.rank(axis=1, ascending=False, method="first")  # 1 = best
    return px, score, rank, eligible, exit_flag


def rebalance_days(index, cfg):
    """Signal dates (the trade executes on the next trading day)."""
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    if cfg.rebalance == "monthly":
        days = s.groupby(idx.to_period("M")).first()
    else:
        wk = s[idx.weekday >= cfg.weekday].groupby(idx[idx.weekday >= cfg.weekday].to_period("W")).first()
        days = wk.iloc[::2] if cfg.rebalance == "biweekly" else wk
    return set(days.values)


# --------------------------------------------------------------------------
# Backtest engine
# --------------------------------------------------------------------------
def category_ranks(score, cats):
    """Rank of each ETF within its category (1 = best score in that category)."""
    out = pd.DataFrame(index=score.index, columns=score.columns, dtype=float)
    groups = {}
    for t in score.columns:
        groups.setdefault(cats.get(t, t), []).append(t)
    for members in groups.values():
        out[members] = score[members].rank(axis=1, ascending=False, method="first")
    return out


def pick_candidates(score_row, elig_row, held, prices, slots, cfg, cats):
    """Best eligible ETFs not already held; with ETF Group on, cap holdings per category."""
    order = score_row[elig_row].sort_values(ascending=False).index
    count = {}
    for t in held:
        count[cats.get(t, t)] = count.get(cats.get(t, t), 0) + 1
    picks = []
    for t in order:
        if len(picks) >= slots:
            break
        if t in held or pd.isna(prices[t]):
            continue
        c = cats.get(t, t)
        if cfg.max_per_category and count.get(c, 0) >= cfg.max_per_category:
            continue
        picks.append(t)
        count[c] = count.get(c, 0) + 1
    return picks


def run_backtest(close, cfg):
    px, score, rank, eligible, exit_flag = compute_signals(close, cfg)
    dates = px.index[(px.index >= pd.Timestamp(cfg.start))]
    if cfg.end:
        dates = dates[dates <= pd.Timestamp(cfg.end)]
    if len(dates) < 30:
        sys.exit("Not enough data in the chosen date range.")
    cs = cash_series(close)
    cash_ret = (cs.pct_change(fill_method=None).fillna(0)
                if cfg.use_cash_etf and cs is not None
                else pd.Series(0.0, index=close.index))
    signal_days = rebalance_days(dates, cfg)
    cost = cfg.cost_bps / 10_000
    cats = category_map(cfg)
    cat_rank = category_ranks(score, cats)

    cash = cfg.capital
    pos = {}            # ticker -> dict(shares, entry_px, entry_date, cost_in)
    trades, equity = [], []
    pending_signal = None

    def sell(t, price, day, reason):
        nonlocal cash
        p = pos.pop(t)
        gross = p["shares"] * price
        cash += gross * (1 - cost)
        pnl = gross * (1 - cost) - p["cost_in"]
        trades.append(dict(Ticker=t, Group=UNIVERSE.get(t, ""), Category=cats.get(t, ""),
                           EntryDate=p["entry_date"].date(), EntryPrice=round(p["entry_px"], 4),
                           ExitDate=day.date(), ExitPrice=round(price, 4),
                           Shares=round(p["shares"], 4), PnL=round(pnl, 2),
                           ReturnPct=round(pnl / p["cost_in"] * 100, 2),
                           DaysHeld=(day - p["entry_date"]).days, ExitReason=reason))

    for i, day in enumerate(dates):
        prices = px.loc[day]
        # 1) cash sleeve earns T-bill return
        cash *= 1 + float(cash_ret.get(day, 0.0))

        # 2) daily stop-loss check (at the close)
        if cfg.stop_loss:
            for t in list(pos):
                p_now = prices[t]
                if pd.notna(p_now) and p_now <= pos[t]["entry_px"] * (1 - cfg.stop_loss):
                    sell(t, p_now, day, "Stop Loss")

        # 3) execute yesterday's rebalance signal at today's close
        if pending_signal is not None:
            sd = pending_signal
            r, elig, xf = rank.loc[sd], eligible.loc[sd], exit_flag.loc[sd]
            for t in list(pos):
                if pd.isna(prices[t]):
                    continue
                if pd.isna(r[t]) or r[t] > cfg.exit_rank:
                    sell(t, prices[t], day, "Rank Exit")
                elif xf[t]:
                    sell(t, prices[t], day, f"Exit: {cfg.exit_condition}")
                elif cfg.max_per_category and cat_rank.loc[sd, t] > cfg.max_per_category + cfg.category_buffer:
                    sell(t, prices[t], day, "ETF Group Swap")
            slots = cfg.n_holdings - len(pos)
            if slots > 0:
                cands = pick_candidates(score.loc[sd], elig, pos, prices, slots, cfg, cats)
                total = cash + sum(p["shares"] * prices[t] for t, p in pos.items()
                                   if pd.notna(prices[t]))
                target = total / cfg.n_holdings
                for t in cands:
                    spend = min(target, cash)
                    if spend <= 1:
                        break
                    sh = spend * (1 - cost) / prices[t]
                    pos[t] = dict(shares=sh, entry_px=prices[t], entry_date=day, cost_in=spend)
                    cash -= spend
            pending_signal = None
        if day in signal_days:
            pending_signal = day

        held_val = sum(p["shares"] * (prices[t] if pd.notna(prices[t]) else p["entry_px"])
                       for t, p in pos.items())
        equity.append(dict(Date=day, Equity=cash + held_val, Cash=cash,
                           Holdings=",".join(sorted(pos))))

    last = dates[-1]
    for t in list(pos):
        sell(t, px.loc[last, t], last, "End of Period")
    eq = pd.DataFrame(equity).set_index("Date")
    return eq, pd.DataFrame(trades)


# --------------------------------------------------------------------------
# Metrics and reports
# --------------------------------------------------------------------------
def perf_stats(equity):
    r = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = r.std() * math.sqrt(252)
    sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else np.nan
    downside = r[r < 0].std() * math.sqrt(252)
    sortino = r.mean() * 252 / downside if downside > 0 else np.nan
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    underwater = (dd < 0).astype(int)
    longest = underwater.groupby((underwater == 0).cumsum()).sum().max()
    return dict(CAGR=cagr, Volatility=vol, Sharpe=sharpe, Sortino=sortino,
                MaxDD=max_dd, Calmar=cagr / abs(max_dd) if max_dd < 0 else np.nan,
                LongestDrawdownDays=int(longest), Years=years)


def benchmarks(close, eq_index, capital):
    out = {}
    if BENCHMARK in close.columns:
        spy = close[BENCHMARK].reindex(eq_index).ffill()
        out[bench_label()] = spy / spy.iloc[0] * capital
    if BENCHMARK in close.columns and BOND_FOR_6040 in close.columns:
        r = close[[BENCHMARK, BOND_FOR_6040]].reindex(eq_index).ffill().pct_change().fillna(0)
        val, vals, w = capital, [], np.array([0.6, 0.4])
        hold = w * val
        month = eq_index[0].month
        for d, row in r.iterrows():
            hold = hold * (1 + row.values)
            val = hold.sum()
            if d.month != month:          # monthly rebalance back to 60/40
                hold, month = w * val, d.month
            vals.append(val)
        out[f"60/40 {bench_name()}/{BOND_FOR_6040}"] = pd.Series(vals, index=eq_index)
    return out


def monthly_table(equity):
    m = equity.resample("ME").last().pct_change()
    m.iloc[0] = equity.resample("ME").last().iloc[0] / equity.iloc[0] - 1
    t = pd.DataFrame({"Year": m.index.year, "Month": m.index.strftime("%b"), "Ret": m.values})
    table = t.pivot(index="Year", columns="Month", values="Ret")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table = table.reindex(columns=[c for c in order if c in table.columns])
    yearly = equity.resample("YE").last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / equity.iloc[0] - 1
    table["Year Total"] = yr.values
    return (table * 100).round(2)


def report(cfg, eq, trades, close, out_root="results", quiet=False):
    out = os.path.join(out_root, cfg.name)
    os.makedirs(out, exist_ok=True)
    stats = perf_stats(eq["Equity"])
    bm = benchmarks(close, eq.index, cfg.capital)
    bm_stats = {k: perf_stats(v) for k, v in bm.items()}

    t = trades
    wins, losses = t[t.PnL > 0], t[t.PnL <= 0]
    months = max(stats["Years"] * 12, 1)
    trade_stats = dict(
        Trades=len(t), WinRate=len(wins) / len(t) if len(t) else np.nan,
        AvgWin=wins.PnL.mean() if len(wins) else 0, AvgLoss=losses.PnL.mean() if len(losses) else 0,
        AvgDaysHeld=t.DaysHeld.mean() if len(t) else 0, TradesPerMonth=len(t) / months)

    eq.to_csv(os.path.join(out, "equity.csv"))
    t.to_csv(os.path.join(out, "trades.csv"), index=False)
    mt = monthly_table(eq["Equity"])
    mt.to_csv(os.path.join(out, "monthly_returns.csv"))
    by_group = t.groupby("Group").agg(Trades=("PnL", "size"), NetPnL=("PnL", "sum")).sort_values("NetPnL", ascending=False)
    by_ticker = t.groupby("Ticker").agg(Trades=("PnL", "size"), NetPnL=("PnL", "sum")).sort_values("NetPnL", ascending=False)
    by_reason = t.groupby("ExitReason").agg(Trades=("PnL", "size"), NetPnL=("PnL", "sum"))

    lines = []
    lines.append(f"ETF Momentum Backtest - {cfg.name}")
    lines.append("=" * 60)
    lines.append(f"Period     : {eq.index[0].date()} to {eq.index[-1].date()} ({stats['Years']:.1f} years)")
    lines.append(f"Settings   : top {cfg.n_holdings}, exit rank {cfg.exit_rank}, rank={cfg.rank_criteria} "
                 f"{cfg.periods}/{cfg.weights}, rebalance={cfg.rebalance}, stop={cfg.stop_loss}, "
                 f"RS={cfg.rs_rule if cfg.use_rs else 'off'}, entry={cfg.entry_condition}, exit={cfg.exit_condition}, retracement={cfg.retracement}, cost={cfg.cost_bps}bps/side, "
                 f"ETF group={'max ' + str(cfg.max_per_category) + ' per category' if cfg.max_per_category else 'off'}")
    if cfg.exclude:
        lines.append(f"Excluded   : {', '.join(cfg.exclude)}")
    lines.append(f"Capital    : {CCY}{cfg.capital:,.0f} -> {CCY}{eq['Equity'].iloc[-1]:,.0f}")
    lines.append("")
    hdr = f"{'':22}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>8}{'Calmar':>8}"
    lines.append(hdr)
    for name, s in [("Strategy", stats)] + list(bm_stats.items()):
        lines.append(f"{name:22}{s['CAGR']*100:7.2f}%{s['Volatility']*100:7.2f}%{s['Sharpe']:8.2f}"
                     f"{s['Sortino']:9.2f}{s['MaxDD']*100:7.2f}%{s['Calmar']:8.2f}")
    lines.append(f"\nLongest drawdown: {stats['LongestDrawdownDays']} trading days")
    lines.append(f"Trades: {trade_stats['Trades']}  |  Win rate: {trade_stats['WinRate']*100:.1f}%  |  "
                 f"Avg win {CCY}{trade_stats['AvgWin']:,.0f} / avg loss {CCY}{trade_stats['AvgLoss']:,.0f}  |  "
                 f"Avg hold {trade_stats['AvgDaysHeld']:.0f} days  |  {trade_stats['TradesPerMonth']:.1f} trades/month")
    lines.append("\nP&L by group:\n" + by_group.round(0).to_string())
    if "Category" in t.columns and len(t):
        by_cat = t.groupby("Category").agg(Trades=("PnL", "size"), NetPnL=("PnL", "sum")).sort_values("NetPnL", ascending=False)
        lines.append("\nP&L by category:\n" + by_cat.round(0).to_string())
    lines.append("\nP&L by exit reason:\n" + by_reason.round(0).to_string())
    lines.append("\nTop 5 / bottom 5 tickers:\n" + pd.concat([by_ticker.head(5), by_ticker.tail(5)]).round(0).to_string())
    lines.append("\nYearly returns (%):\n" + mt["Year Total"].to_string())
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    if not quiet:
        print(text)
        print(f"\nFiles saved in: {os.path.abspath(out)}")
    plot(eq, bm, cfg, out)
    return dict(Name=cfg.name, **{k: stats[k] for k in ["CAGR", "Volatility", "Sharpe", "Sortino", "MaxDD", "Calmar"]},
                Trades=trade_stats["Trades"], WinRate=trade_stats["WinRate"],
                TradesPerMonth=trade_stats["TradesPerMonth"])


def plot(eq, bm, cfg, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(eq.index, eq["Equity"], label="Strategy", color="#1f5fbf", lw=1.6)
    for (name, s), c in zip(bm.items(), ["#999999", "#c9a227"]):
        a1.plot(s.index, s, label=name, color=c, lw=1)
    a1.set_yscale("log")
    a1.set_title(f"US ETF Momentum - {cfg.name} (log scale)")
    a1.legend(loc="upper left")
    a1.grid(alpha=.3)
    dd = eq["Equity"] / eq["Equity"].cummax() - 1
    a2.fill_between(dd.index, dd * 100, 0, color="#c0392b", alpha=.5)
    a2.set_ylabel("Drawdown %")
    a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "equity_curve.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Current signal (what to hold today)
# --------------------------------------------------------------------------
def next_rebalance_preview(close, cfg, held):
    """What the strategy would do if it rebalanced on the latest close, given current holdings.
    Returns (sells: list of (ticker, reason), buys: list of tickers)."""
    px, score, rank, eligible, exit_flag = compute_signals(close, cfg)
    d = px.index[-1]
    cats = category_map(cfg)
    cat_rank = category_ranks(score, cats)
    sells, keep = [], []
    for t in held:
        if t not in px.columns:
            sells.append((t, "Not in universe"))
        elif pd.isna(rank.loc[d, t]) or rank.loc[d, t] > cfg.exit_rank:
            sells.append((t, f"Rank Exit (rank {rank.loc[d, t]:.0f})" if pd.notna(rank.loc[d, t]) else "Rank Exit"))
        elif exit_flag.loc[d, t]:
            sells.append((t, f"Exit: {cfg.exit_condition}"))
        elif cfg.max_per_category and cat_rank.loc[d, t] > cfg.max_per_category + cfg.category_buffer:
            sells.append((t, "ETF Group Swap"))
        else:
            keep.append(t)
    buys = pick_candidates(score.loc[d], eligible.loc[d], {t: None for t in keep}, px.loc[d],
                           cfg.n_holdings - len(keep), cfg, cats)
    return d, sells, buys, keep


def current_signal(close, cfg):
    px, score, rank, eligible, exit_flag = compute_signals(close, cfg)
    d = px.index[-1]
    cats = category_map(cfg)
    df = pd.DataFrame({"Group": [UNIVERSE[t] for t in px.columns],
                       "Category": [cats.get(t, "") for t in px.columns], "Score": score.loc[d],
                       "Rank": rank.loc[d], "Close": px.loc[d], "ExitSignal": exit_flag.loc[d],
                       "Eligible": eligible.loc[d]}).sort_values("Rank")
    print(f"\nSignal as of {d.date()} (top 15 by score):")
    print(df.head(15).round(3).to_string())
    buys = pick_candidates(score.loc[d], eligible.loc[d], {}, px.loc[d], cfg.n_holdings, cfg, cats)
    print(f"\nWould hold: {buys if buys else 'none - all in ' + CASH_ETF}")
    if len(buys) < cfg.n_holdings:
        print(f"{cfg.n_holdings - len(buys)} slot(s) in {CASH_ETF}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):      # allow ₹ etc. in Windows console/log files
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="US ETF momentum rotation backtester")
    ap.add_argument("--market", choices=list(PROFILES), default="us",
                    help="us = US-listed ETFs; ireland = Ireland-domiciled UCITS ETFs on LSE")
    ap.add_argument("--start", default=None, help="default 2007-01-01 (us) / 2011-01-01 (ireland)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--n", type=int, default=5, help="number of holdings")
    ap.add_argument("--exit-rank", type=int, default=10)
    ap.add_argument("--rank", choices=["sharpe", "return"], default="sharpe")
    ap.add_argument("--rebalance", choices=["weekly", "biweekly", "monthly"], default="weekly")
    ap.add_argument("--stop", type=float, default=0.12, help="stop loss, e.g. 0.12; 0 = off")
    ap.add_argument("--no-rs", action="store_true", help="turn off relative-strength filter")
    ap.add_argument("--no-retracement", action="store_true")
    ap.add_argument("--no-cash-etf", action="store_true", help="hold idle cash instead of BIL")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--exclude", default="", help="comma list, e.g. GLD,SLV")
    ap.add_argument("--only", default="", help="comma list: test ONLY these ETFs, e.g. SPY,QQQ,GLD,TLT")
    ap.add_argument("--entry", default="Close above 200 EMA", choices=list(ENTRY_CONDITIONS))
    ap.add_argument("--exit", dest="exit_cond", default="Close below 200 EMA", choices=list(EXIT_CONDITIONS))
    ap.add_argument("--rs-rule", default="Ratio above 200 EMA", choices=list(RS_RULES))
    ap.add_argument("--etf-group", type=int, default=0,
                    help="max ETFs per category (1 = only the best per category; 0 = off)")
    ap.add_argument("--name", default=None, help="results folder name")
    ap.add_argument("--sweep", action="store_true", help="run the comparison set")
    ap.add_argument("--signal", action="store_true", help="print today's holdings signal")
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--csv", default=None, help="use your own price CSV (date index, ticker columns)")
    a = ap.parse_args()

    set_profile(a.market)
    close = load_prices(refresh=a.refresh, csv_path=a.csv)
    if a.start is None:
        a.start = PROFILES[a.market]["default_start"]
    cfg = Config(start=a.start, end=a.end, capital=a.capital, n_holdings=a.n,
                 exit_rank=a.exit_rank, rank_criteria=a.rank, rebalance=a.rebalance,
                 stop_loss=a.stop or None, use_rs=not a.no_rs,
                 retracement=None if a.no_retracement else 0.20,
                 use_cash_etf=not a.no_cash_etf, cost_bps=a.cost_bps,
                 exclude=tuple(x.strip().upper() for x in a.exclude.split(",") if x.strip()),
                 max_per_category=a.etf_group,
                 include=tuple(x.strip().upper() for x in a.only.split(",") if x.strip()), entry_condition=a.entry,
                 exit_condition=a.exit_cond, rs_rule=a.rs_rule)
    cfg.name = a.name or ("base" if a.market == "us" else f"{a.market}_base")

    if a.signal:
        current_signal(close, cfg)
        return

    if not a.sweep:
        eq, trades = run_backtest(close, cfg)
        report(cfg, eq, trades, close)
        return

    variants = [
        replace(cfg, name="01_base_weekly"),
        replace(cfg, name="02_monthly", rebalance="monthly"),
        replace(cfg, name="03_biweekly", rebalance="biweekly"),
        replace(cfg, name="04_no_stop", stop_loss=None),
        replace(cfg, name="05_stop_08", stop_loss=0.08),
        replace(cfg, name="06_stop_10", stop_loss=0.10),
        replace(cfg, name="07_stop_15", stop_loss=0.15),
        replace(cfg, name="08_no_rs_filter", use_rs=False),
        replace(cfg, name="09_ex_gold_silver", exclude=tuple(set(cfg.exclude) | set(GOLD_SILVER))),
        replace(cfg, name="10_rank_by_return", rank_criteria="return"),
        replace(cfg, name="11_monthly_no_rs", rebalance="monthly", use_rs=False),
        replace(cfg, name="12_etf_group_on", max_per_category=1),
        replace(cfg, name="13_etf_group_off", max_per_category=0),
    ]
    sweep_dir = "results/sweep" if a.market == "us" else f"results/sweep_{a.market}"
    rows = []
    for v in variants:
        print(f"Running {v.name} ...")
        eq, trades = run_backtest(close, v)
        rows.append(report(v, eq, trades, close, out_root=sweep_dir, quiet=True))
    res = pd.DataFrame(rows).set_index("Name")
    pct = ["CAGR", "Volatility", "MaxDD", "WinRate"]
    show = res.copy()
    show[pct] = (show[pct] * 100).round(2)
    show = show.round(2)
    show.to_csv(os.path.join(sweep_dir, "comparison.csv"))
    print("\nSweep comparison (CAGR / Vol / MaxDD / WinRate in %):")
    print(show.to_string())
    print(f"\nSaved: {os.path.abspath(os.path.join(sweep_dir, 'comparison.csv'))}")


if __name__ == "__main__":
    main()
