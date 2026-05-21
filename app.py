import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import shapiro, linregress

st.set_page_config(page_title="S&P 500 Long + Tactical Hedge Lab", layout="wide")

st.title("S&P 500 Long + Tactical SPY Hedge Lab")

# ============================================================
# USER CONTROLS
# ============================================================

st.sidebar.header("Strategy settings")
st.sidebar.info("On mobile, tap the » icon at top-left to open these controls.")

PRICE_PERIOD = st.sidebar.selectbox(
    "Backtest period",
    ["3y", "5y", "7y", "10y"],
    index=1,
)

LOOKBACK = st.sidebar.slider("Normality/trend lookback days", 20, 60, 30, 5)

P_THRESHOLD = st.sidebar.slider(
    "Normality p-value threshold",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
)

MAX_LONGS = st.sidebar.slider(
    "Maximum number of long positions",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
)

MIN_HOLD_DAYS = st.sidebar.slider(
    "Minimum holding period, days",
    min_value=0,
    max_value=30,
    value=3,
    step=1,
)

rebalance_choice = st.sidebar.selectbox(
    "Rebalance frequency",
    ["Daily", "Monday only"],
    index=0,
)

REBALANCE_WEEKDAY = None if rebalance_choice == "Daily" else 0

TREND_EXIT_THRESHOLD = st.sidebar.number_input(
    "Trend exit threshold",
    value=-0.01,
    step=0.01,
    format="%.3f",
)

IMPLIED_SCORE_EXIT_THRESHOLD = st.sidebar.number_input(
    "Implied revision exit threshold",
    value=-5.0,
    step=1.0,
    format="%.1f",
)

st.sidebar.header("Trading costs")

STOCK_COST_BPS = st.sidebar.slider(
    "Stock cost per side, bps",
    min_value=0,
    max_value=100,
    value=10,
    step=1,
)

HEDGE_COST_BPS = st.sidebar.slider(
    "SPY hedge cost per side, bps",
    min_value=0,
    max_value=50,
    value=2,
    step=1,
)

STOCK_COST_PER_SIDE = STOCK_COST_BPS / 10000
HEDGE_COST_PER_SIDE = HEDGE_COST_BPS / 10000

st.sidebar.header("Hedge settings")

HEDGE_SIZE = st.sidebar.slider(
    "SPY hedge size",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
)

NFCI_LOOSE_Z = st.sidebar.slider(
    "Loose conditions z trigger",
    min_value=0.0,
    max_value=3.0,
    value=0.75,
    step=0.05,
)

SKEW_LOW_Z = st.sidebar.slider(
    "Low SKEW z trigger",
    min_value=-3.0,
    max_value=0.0,
    value=-0.75,
    step=0.05,
)

MOMENTUM_EUPHORIA_Z = st.sidebar.slider(
    "SPY momentum z trigger",
    min_value=0.0,
    max_value=3.0,
    value=1.00,
    step=0.05,
)

VIX_LOW_Z = st.sidebar.slider(
    "Low VIX z trigger",
    min_value=-3.0,
    max_value=0.0,
    value=-0.50,
    step=0.05,
)

REQUIRE_MARKET_CRACK = st.sidebar.checkbox(
    "Require SPY crack before hedge activates",
    value=True,
)

st.sidebar.caption(
    "Hedge logic: loose financial conditions + complacency/euphoria + optional SPY crack. "
    "The crack filter tries to avoid shorting too early in melt-up markets."
)

# DRAM / memory-cycle proxy weights
MEMORY_WEIGHT_TECH = 20
MEMORY_WEIGHT_COMMUNICATIONS = 8
MEMORY_WEIGHT_INDUSTRIALS = 5

st.caption(
    "Systematic research tool only, not investment advice. "
    "Long book uses return-distribution normalisation, trend persistence, relative strength, "
    "macro/earnings-revision pressure and a DRAM/memory-cycle proxy. "
    "The hedge shorts SPY when financial conditions are loose and equity complacency is high, "
    "but now optionally requires evidence that SPY is starting to crack."
)

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=60 * 60 * 12)
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=20).text
    table = pd.read_html(StringIO(html))[0]
    table["Ticker"] = table["Symbol"].str.replace(".", "-", regex=False)
    return table[["Ticker", "Security", "GICS Sector"]]


@st.cache_data(ttl=60 * 60 * 6)
def get_prices(tickers, period):
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    return close.dropna(axis=1, how="all").ffill()


MARKET_TICKERS = {
    "Market": "SPY",
    "Credit": "HYG",
    "Rates": "TLT",
    "Dollar": "UUP",
    "Oil ETF": "USO",
    "Semiconductors": "SOXX",
    "Memory Proxy": "MU",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


@st.cache_data(ttl=60 * 60 * 6)
def get_market_proxy_prices(period):
    data = yf.download(
        list(MARKET_TICKERS.values()),
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    return close.dropna(axis=1, how="all").ffill()


@st.cache_data(ttl=60 * 60 * 6)
def get_market_sentiment_data(period):
    tickers = {"SPY": "SPY", "VIX": "^VIX", "SKEW": "^SKEW"}
    out = {}

    for name, ticker in tickers.items():
        try:
            data = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if data is None or data.empty:
                continue

            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.dropna()

            if len(close) > 50:
                out[name] = close

        except Exception:
            pass

    return out


FRED_SERIES = {
    "industrial_production": "INDPRO",
    "retail_sales": "RSAFS",
    "payrolls": "PAYEMS",
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "ppi": "PPIACO",
    "ten_year": "DGS10",
    "two_year": "DGS2",
    "hy_spread": "BAMLH0A0HYM2",
    "financial_conditions": "NFCI",
    "m2": "M2SL",
    "dollar": "DTWEXBGS",
    "oil": "DCOILWTICO",
}


@st.cache_data(ttl=60 * 60 * 12)
def get_fred_data(period):
    end = pd.Timestamp.today()
    years = int(period.replace("y", "")) if period.endswith("y") else 5
    start = end - pd.DateOffset(years=years)
    series = {}

    for name, code in FRED_SERIES.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
            raw = pd.read_csv(url)
            raw.columns = ["Date", name]
            raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
            raw[name] = pd.to_numeric(raw[name].replace(".", np.nan), errors="coerce")
            raw = raw.dropna(subset=["Date"])
            raw = raw[(raw["Date"] >= start) & (raw["Date"] <= end)]
            s = raw.set_index("Date")[name].dropna()
            s = s.resample("ME").last()
            series[name] = s
        except Exception:
            series[name] = pd.Series(dtype=float)

    return pd.DataFrame(series).sort_index().ffill()


# ============================================================
# HELPERS
# ============================================================

def trend_score(series):
    series = series.dropna()
    if len(series) < 3:
        return np.nan

    y = np.log(series.values)
    x = np.arange(len(y))
    return linregress(x, y).slope * 100


def normality_pvalue(series, lookback):
    returns = series.pct_change().dropna()
    if len(returns) < lookback - 1:
        return np.nan

    try:
        return shapiro(returns).pvalue
    except Exception:
        return np.nan


def pct_return(price_df, ticker, days=30):
    try:
        if ticker is None or ticker not in price_df.columns:
            return 0.0

        s = price_df[ticker].dropna()

        if len(s) < days + 1:
            return 0.0

        return s.iloc[-1] / s.iloc[-days] - 1

    except Exception:
        return 0.0


def realised_vol(series):
    r = series.pct_change().dropna()
    return r.std() if len(r) >= 3 else 0.0


def vol_compression_for_ticker(ticker, prices, lookback):
    try:
        s = prices[ticker].dropna()
        if len(s) < lookback:
            return 0.0

        return realised_vol(s.iloc[-30:-15]) - realised_vol(s.iloc[-15:])

    except Exception:
        return 0.0


def latest_zscore(series, window=252):
    s = series.dropna()
    if len(s) < max(30, window // 2):
        return np.nan

    w = s.iloc[-window:] if len(s) >= window else s
    std = w.std()

    if std == 0 or np.isnan(std):
        return np.nan

    return float((w.iloc[-1] - w.mean()) / std)


def zscore_latest(series, lookback=36, transform="diff"):
    s = series.dropna()
    if len(s) < 15:
        return 0.0

    if transform == "yoy":
        x = s.pct_change(12).dropna()
    elif transform == "diff":
        x = s.diff(3).dropna()
    elif transform == "level":
        x = s.copy()
    else:
        x = s.copy()

    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) < 12:
        return 0.0

    window = x.iloc[-lookback:] if len(x) >= lookback else x
    std = window.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return float((window.iloc[-1] - window.mean()) / std)


def same_or_before(series, date):
    try:
        return series[series.index <= date]
    except Exception:
        return pd.Series(dtype=float)


# ============================================================
# MACRO MODEL
# ============================================================

def build_fred_macro_factor_scores(fred):
    if fred.empty:
        return {
            "Growth": 0.0,
            "Inflation relief": 0.0,
            "Financial conditions": 0.0,
            "Liquidity": 0.0,
            "Dollar relief": 0.0,
            "Oil": 0.0,
            "Yield curve": 0.0,
            "Loose conditions": 0.0,
        }

    growth = np.nanmean([
        zscore_latest(fred.get("industrial_production", pd.Series(dtype=float)), transform="yoy"),
        zscore_latest(fred.get("retail_sales", pd.Series(dtype=float)), transform="yoy"),
        zscore_latest(fred.get("payrolls", pd.Series(dtype=float)), transform="diff"),
        -zscore_latest(fred.get("unemployment", pd.Series(dtype=float)), transform="diff"),
    ])

    inflation_relief = -np.nanmean([
        zscore_latest(fred.get("cpi", pd.Series(dtype=float)), transform="diff"),
        zscore_latest(fred.get("ppi", pd.Series(dtype=float)), transform="diff"),
    ])

    financial_conditions = np.nanmean([
        -zscore_latest(fred.get("hy_spread", pd.Series(dtype=float)), transform="level"),
        -zscore_latest(fred.get("financial_conditions", pd.Series(dtype=float)), transform="level"),
    ])

    loose_conditions = -zscore_latest(
        fred.get("financial_conditions", pd.Series(dtype=float)),
        transform="level",
    )

    liquidity = zscore_latest(fred.get("m2", pd.Series(dtype=float)), transform="yoy")
    dollar_relief = -zscore_latest(fred.get("dollar", pd.Series(dtype=float)), transform="diff")
    oil = zscore_latest(fred.get("oil", pd.Series(dtype=float)), transform="diff")

    try:
        curve = (fred["ten_year"] - fred["two_year"]).dropna()
        yield_curve = zscore_latest(curve, transform="level")
    except Exception:
        yield_curve = 0.0

    return {
        "Growth": float(np.nan_to_num(growth)),
        "Inflation relief": float(np.nan_to_num(inflation_relief)),
        "Financial conditions": float(np.nan_to_num(financial_conditions)),
        "Liquidity": float(np.nan_to_num(liquidity)),
        "Dollar relief": float(np.nan_to_num(dollar_relief)),
        "Oil": float(np.nan_to_num(oil)),
        "Yield curve": float(np.nan_to_num(yield_curve)),
        "Loose conditions": float(np.nan_to_num(loose_conditions)),
    }


SECTOR_FRED_WEIGHTS = {
    "Information Technology": {
        "Growth": 0.25,
        "Inflation relief": 0.20,
        "Financial conditions": 0.20,
        "Liquidity": 0.25,
        "Dollar relief": 0.10,
    },
    "Communication Services": {
        "Growth": 0.30,
        "Inflation relief": 0.15,
        "Financial conditions": 0.15,
        "Liquidity": 0.20,
        "Dollar relief": 0.20,
    },
    "Consumer Discretionary": {
        "Growth": 0.35,
        "Inflation relief": 0.20,
        "Financial conditions": 0.20,
        "Liquidity": 0.15,
        "Dollar relief": 0.10,
    },
    "Consumer Staples": {
        "Growth": 0.10,
        "Inflation relief": 0.35,
        "Financial conditions": 0.15,
        "Liquidity": 0.10,
        "Dollar relief": 0.30,
    },
    "Industrials": {
        "Growth": 0.45,
        "Inflation relief": 0.10,
        "Financial conditions": 0.20,
        "Liquidity": 0.10,
        "Dollar relief": 0.15,
    },
    "Materials": {
        "Growth": 0.45,
        "Inflation relief": 0.05,
        "Financial conditions": 0.15,
        "Liquidity": 0.10,
        "Dollar relief": 0.15,
        "Oil": 0.10,
    },
    "Energy": {
        "Growth": 0.15,
        "Inflation relief": -0.10,
        "Financial conditions": 0.10,
        "Liquidity": 0.05,
        "Dollar relief": 0.10,
        "Oil": 0.70,
    },
    "Financials": {
        "Growth": 0.25,
        "Inflation relief": 0.05,
        "Financial conditions": 0.25,
        "Liquidity": 0.05,
        "Yield curve": 0.40,
    },
    "Health Care": {
        "Growth": 0.10,
        "Inflation relief": 0.25,
        "Financial conditions": 0.15,
        "Liquidity": 0.15,
        "Dollar relief": 0.35,
    },
    "Real Estate": {
        "Growth": 0.10,
        "Inflation relief": 0.30,
        "Financial conditions": 0.25,
        "Liquidity": 0.25,
        "Yield curve": -0.10,
    },
    "Utilities": {
        "Growth": -0.05,
        "Inflation relief": 0.35,
        "Financial conditions": 0.25,
        "Liquidity": 0.25,
        "Yield curve": -0.10,
    },
}


def fred_sector_macro_score(sector, fred_factor_scores):
    weights = SECTOR_FRED_WEIGHTS.get(
        sector,
        {
            "Growth": 0.30,
            "Inflation relief": 0.20,
            "Financial conditions": 0.20,
            "Liquidity": 0.15,
            "Dollar relief": 0.15,
        },
    )

    return sum(weights.get(k, 0.0) * fred_factor_scores.get(k, 0.0) for k in weights)
    
def memory_cycle_momentum(market_prices):
    mu_mom = pct_return(market_prices, "MU")
    soxx_mom = pct_return(market_prices, "SOXX")

    if mu_mom == 0 and soxx_mom == 0:
        return 0.0
    if mu_mom == 0:
        return soxx_mom
    if soxx_mom == 0:
        return mu_mom

    return 0.65 * mu_mom + 0.35 * soxx_mom


def memory_factor_contribution(sector, market_prices):
    mem = memory_cycle_momentum(market_prices)

    if sector == "Information Technology":
        return MEMORY_WEIGHT_TECH * mem
    if sector == "Communication Services":
        return MEMORY_WEIGHT_COMMUNICATIONS * mem
    if sector == "Industrials":
        return MEMORY_WEIGHT_INDUSTRIALS * mem

    return 0.0


def market_macro_score(sector, ticker, prices, market_prices):
    sector_etf = MARKET_TICKERS.get(sector)

    score = (
        45 * pct_return(market_prices, sector_etf)
        + 25 * pct_return(market_prices, "SPY")
        + 20 * pct_return(market_prices, "HYG")
        - 10 * pct_return(market_prices, "TLT")
        - 10 * pct_return(market_prices, "UUP")
    )

    if sector == "Energy":
        score += 25 * pct_return(market_prices, "USO")

    score += memory_factor_contribution(sector, market_prices)

    try:
        stock_ret = pct_return(prices, ticker)
        sector_ret = pct_return(market_prices, sector_etf)
        relative_strength = (stock_ret - sector_ret) * 100
    except Exception:
        relative_strength = 0.0

    return score + relative_strength


def combined_macro_earnings_score(sector, ticker, prices, market_prices, fred_factor_scores):
    fred_score = fred_sector_macro_score(sector, fred_factor_scores)
    mkt_score = market_macro_score(sector, ticker, prices, market_prices)
    total = 2.0 * fred_score + 0.35 * mkt_score
    return total, fred_score, mkt_score


def implied_earnings_revision_score(row):
    return (
        40 * row["Relative strength vs sector"]
        + 25 * row["Relative strength vs market"]
        + 20 * row["Trend score"]
        + 15 * row["Vol compression"]
        + 10 * row["Combined macro earnings score"]
    )


def hedge_signal_from_data(
    market_prices,
    fred_factor_scores,
    sentiment=None,
    hedge_size=0.50,
    nfci_loose_z=0.75,
    skew_low_z=-0.75,
    momentum_euphoria_z=1.00,
    vix_low_z=-0.50,
    require_market_crack=True,
):
    loose_z = fred_factor_scores.get("Loose conditions", 0.0)

    skew_z = np.nan
    vix_z = np.nan
    spy_mom_z = np.nan
    spy_crack_flag = False

    if sentiment is not None and "SKEW" in sentiment:
        skew_z = latest_zscore(sentiment["SKEW"], 252)

    if sentiment is not None and "VIX" in sentiment:
        vix_z = latest_zscore(sentiment["VIX"], 252)

    if sentiment is not None and "SPY" in sentiment:
        spy = sentiment["SPY"].dropna()

        if len(spy) > 63:
            spy_3m = spy.pct_change(63).dropna()
            spy_mom_z = latest_zscore(spy_3m, 252)

        if len(spy) > 60:
            spy_50dma = spy.rolling(50).mean().iloc[-1]
            spy_below_50dma = spy.iloc[-1] < spy_50dma
            spy_20d_return = spy.iloc[-1] / spy.iloc[-20] - 1
            spy_crack_flag = bool(spy_below_50dma or spy_20d_return < 0)

    low_skew_flag = bool(pd.notna(skew_z) and skew_z < skew_low_z)
    euphoric_momentum_flag = bool(pd.notna(spy_mom_z) and spy_mom_z > momentum_euphoria_z)
    complacent_vix_flag = bool(pd.notna(vix_z) and vix_z < vix_low_z)
    loose_flag = bool(loose_z > nfci_loose_z)

    complacency_flag = bool(
        low_skew_flag
        or euphoric_momentum_flag
        or complacent_vix_flag
    )

    if require_market_crack:
        hedge_on = bool(loose_flag and complacency_flag and spy_crack_flag)
    else:
        hedge_on = bool(loose_flag and complacency_flag)

    return {
        "hedge_on": hedge_on,
        "hedge_size": hedge_size if hedge_on else 0.0,
        "loose_conditions_z": loose_z,
        "skew_z": skew_z,
        "spy_3m_momentum_z": spy_mom_z,
        "vix_z": vix_z,
        "loose_flag": loose_flag,
        "low_skew_flag": low_skew_flag,
        "euphoric_momentum_flag": euphoric_momentum_flag,
        "complacent_vix_flag": complacent_vix_flag,
        "spy_crack_flag": spy_crack_flag,
    }


def hedge_signal_at(
    market_prices_slice,
    fred_factor_scores,
    full_sentiment,
    date,
    hedge_size,
    nfci_loose_z,
    skew_low_z,
    momentum_euphoria_z,
    vix_low_z,
    require_market_crack,
):
    sentiment_slice = {}

    for key, series in full_sentiment.items():
        s = same_or_before(series, date)
        if len(s) > 0:
            sentiment_slice[key] = s

    return hedge_signal_from_data(
        market_prices_slice,
        fred_factor_scores,
        sentiment_slice,
        hedge_size,
        nfci_loose_z,
        skew_low_z,
        momentum_euphoria_z,
        vix_low_z,
        require_market_crack,
    )


def performance_stats(bt):
    bt = bt.dropna().copy()

    if bt.empty:
        return bt, np.nan, np.nan, np.nan, np.nan, np.nan

    bt["Equity curve"] = (1 + bt["Return"]).cumprod()
    bt["Long book equity"] = (1 + bt["Long book return"]).cumprod()
    bt["Hedge equity"] = (1 + bt["Hedge return"]).cumprod()
    bt["SPY equity"] = (1 + bt["SPY return"]).cumprod()

    total_return = bt["Equity curve"].iloc[-1] - 1
    annualised_return = bt["Equity curve"].iloc[-1] ** (252 / len(bt)) - 1
    annualised_vol = bt["Return"].std() * np.sqrt(252)
    sharpe = (
        bt["Return"].mean() / bt["Return"].std() * np.sqrt(252)
        if bt["Return"].std() != 0
        else np.nan
    )

    drawdown = bt["Equity curve"] / bt["Equity curve"].cummax() - 1
    max_drawdown = drawdown.min()

    return bt, total_return, annualised_return, annualised_vol, sharpe, max_drawdown


def show_backtest(name, bt):
    st.markdown(f"### {name}")

    if bt.empty:
        st.write("No results generated.")
        return bt

    bt, total_return, annualised_return, annualised_vol, sharpe, max_drawdown = performance_stats(bt)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total return", f"{total_return:.2%}")
    c2.metric("Annualised return", f"{annualised_return:.2%}")
    c3.metric("Annualised vol", f"{annualised_vol:.2%}")
    c4.metric("Sharpe ratio", f"{sharpe:.2f}")
    c5.metric("Max drawdown", f"{max_drawdown:.2%}")

    st.line_chart(
        bt.set_index("Date")[[
            "Equity curve",
            "Long book equity",
            "Hedge equity",
            "SPY equity",
        ]]
    )

    st.markdown("**Recent portfolio history**")
    st.dataframe(bt.tail(30), use_container_width=True)

    return bt


with st.spinner("Loading S&P 500, prices, market proxies, sentiment and FRED macro data..."):
    sp500 = get_sp500()
    tickers = sp500["Ticker"].tolist()
    prices = get_prices(tickers, PRICE_PERIOD)
    market_prices = get_market_proxy_prices(PRICE_PERIOD)
    sentiment = get_market_sentiment_data(PRICE_PERIOD)
    fred = get_fred_data(PRICE_PERIOD)

fred_factor_scores = build_fred_macro_factor_scores(fred)
available = [t for t in tickers if t in prices.columns]

st.subheader("Macro and hedge dashboard")

fred_table = pd.DataFrame(
    [{"Factor": k, "Latest z-score": v} for k, v in fred_factor_scores.items()]
).sort_values("Factor")

st.dataframe(fred_table, use_container_width=True)

current_hedge = hedge_signal_from_data(
    market_prices,
    fred_factor_scores,
    sentiment,
    HEDGE_SIZE,
    NFCI_LOOSE_Z,
    SKEW_LOW_Z,
    MOMENTUM_EUPHORIA_Z,
    VIX_LOW_Z,
    REQUIRE_MARKET_CRACK,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current hedge active", "Yes" if current_hedge["hedge_on"] else "No")
c2.metric("Hedge size", f"{current_hedge['hedge_size']:.0%}")
c3.metric("Loose conditions z", f"{current_hedge['loose_conditions_z']:.2f}")
c4.metric("SKEW z-score", "n/a" if pd.isna(current_hedge["skew_z"]) else f"{current_hedge['skew_z']:.2f}")

c1, c2, c3 = st.columns(3)
c1.metric("SPY 3m momentum z", "n/a" if pd.isna(current_hedge["spy_3m_momentum_z"]) else f"{current_hedge['spy_3m_momentum_z']:.2f}")
c2.metric("VIX z-score", "n/a" if pd.isna(current_hedge["vix_z"]) else f"{current_hedge['vix_z']:.2f}")
c3.metric("SPY crack flag", "Yes" if current_hedge["spy_crack_flag"] else "No")

c1, c2, c3 = st.columns(3)
c1.metric("Low SKEW flag", "Yes" if current_hedge["low_skew_flag"] else "No")
c2.metric("Low VIX flag", "Yes" if current_hedge["complacent_vix_flag"] else "No")
c3.metric("Momentum euphoria flag", "Yes" if current_hedge["euphoric_momentum_flag"] else "No")

mem_mom_now = memory_cycle_momentum(market_prices)
c1, c2, c3 = st.columns(3)
c1.metric("DRAM / memory proxy 30d", f"{mem_mom_now:.2%}")
c2.metric("MU 30d", f"{pct_return(market_prices, 'MU'):.2%}")
c3.metric("SOXX 30d", f"{pct_return(market_prices, 'SOXX'):.2%}")

rows = []

for ticker in available:
    s = prices[ticker].dropna()

    if len(s) < LOOKBACK + 1:
        continue

    window = s.iloc[-LOOKBACK:]
    pval = normality_pvalue(window, LOOKBACK)
    trend = trend_score(window)
    ret_30d = window.iloc[-1] / window.iloc[0] - 1
    vol_compression = vol_compression_for_ticker(ticker, prices, LOOKBACK)

    if np.isnan(pval) or np.isnan(trend):
        continue

    rows.append({
        "Ticker": ticker,
        "Trend score": trend,
        "30d return": ret_30d,
        "Normality p-value": pval,
        "Pass normality": pval > P_THRESHOLD,
        "Vol compression": vol_compression,
    })

df = pd.DataFrame(rows).merge(sp500, on="Ticker", how="left")

macro_parts = df.apply(
    lambda r: combined_macro_earnings_score(
        r["GICS Sector"],
        r["Ticker"],
        prices,
        market_prices,
        fred_factor_scores,
    ),
    axis=1,
)

df["Combined macro earnings score"] = [x[0] for x in macro_parts]
df["FRED sector macro score"] = [x[1] for x in macro_parts]
df["Market-implied macro score"] = [x[2] for x in macro_parts]
df["Memory factor contribution"] = df["GICS Sector"].apply(
    lambda sector: memory_factor_contribution(sector, market_prices)
)

market_30d_return = pct_return(market_prices, "SPY")

def sector_relative_strength(row):
    sector_etf = MARKET_TICKERS.get(row["GICS Sector"])
    sector_ret = pct_return(market_prices, sector_etf)
    return row["30d return"] - sector_ret

df["Relative strength vs market"] = df["30d return"] - market_30d_return
df["Relative strength vs sector"] = df.apply(sector_relative_strength, axis=1)
df["Implied earnings revision score"] = df.apply(implied_earnings_revision_score, axis=1)

passed = df[df["Pass normality"]]

long_screen = passed[
    (passed["Trend score"] > 0)
    & (passed["30d return"] > 0)
    & (passed["Implied earnings revision score"] > 0)
].sort_values("Implied earnings revision score", ascending=False).head(MAX_LONGS)

st.subheader("Long book: current candidates")

display_cols = [
    "Ticker",
    "Security",
    "GICS Sector",
    "Trend score",
    "30d return",
    "Normality p-value",
    "FRED sector macro score",
    "Market-implied macro score",
    "Combined macro earnings score",
    "Memory factor contribution",
    "Relative strength vs sector",
    "Vol compression",
    "Implied earnings revision score",
]

st.dataframe(long_screen[display_cols], use_container_width=True)

st.subheader("Implied earnings revision overlay")

revision_table = df[[
    "Ticker",
    "Security",
    "GICS Sector",
    "30d return",
    "Relative strength vs market",
    "Relative strength vs sector",
    "Trend score",
    "Vol compression",
    "FRED sector macro score",
    "Market-implied macro score",
    "Combined macro earnings score",
    "Memory factor contribution",
    "Implied earnings revision score",
]].sort_values("Implied earnings revision score", ascending=False)

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Highest implied upward revision pressure")
    st.dataframe(revision_table.head(15), use_container_width=True)

with c2:
    st.markdown("### Lowest implied revision pressure")
    st.dataframe(
        revision_table.tail(15).sort_values("Implied earnings revision score"),
        use_container_width=True,
    )

st.subheader("Backtest")

if not st.button("Run long book + tactical SPY hedge backtest"):
    st.info("Click to run the backtest. First run may take a few minutes.")
    st.stop()


def run_long_with_tactical_hedge_backtest(
    prices,
    sector_table,
    market_prices,
    sentiment,
    fred_factor_scores,
    lookback,
    p_threshold,
    max_longs,
    min_hold_days,
    rebalance_weekday,
    trend_exit_threshold,
    implied_score_exit_threshold,
    stock_cost_per_side,
    hedge_cost_per_side,
    hedge_size,
    nfci_loose_z,
    skew_low_z,
    momentum_euphoria_z,
    vix_low_z,
    require_market_crack,
):
    returns = prices.pct_change()
    spy_returns = market_prices["SPY"].pct_change()

    positions = {}
    holding_days = {}
    results = []
    trade_log = []

    sector_lookup = sector_table.set_index("Ticker")["GICS Sector"].to_dict()
    name_lookup = sector_table.set_index("Ticker")["Security"].to_dict()

    previous_hedge_size = 0.0

    for i in range(lookback + 2, len(prices) - 1):
        signal_date = prices.index[i]
        trade_date = prices.index[i + 1]

        for t in list(holding_days.keys()):
            holding_days[t] += 1

        is_rebalance_day = (
            True
            if rebalance_weekday is None
            else signal_date.weekday() == rebalance_weekday
        )

        entries_today = 0
        exits_today = 0

        if is_rebalance_day:
            for ticker in list(positions.keys()):
                current = prices[ticker].iloc[i - lookback:i].dropna()

                if len(current) < lookback:
                    trade_log.append({
                        "Date": signal_date,
                        "Ticker": ticker,
                        "Security": name_lookup.get(ticker, ""),
                        "Action": "EXIT",
                        "Reason": "Insufficient data",
                    })
                    del positions[ticker]
                    holding_days.pop(ticker, None)
                    exits_today += 1
                    continue

                pval = normality_pvalue(current, lookback)
                trend = trend_score(current)
                ret_30d = current.iloc[-1] / current.iloc[0] - 1

                sector = sector_lookup.get(ticker, None)
                macro_total, fred_score, market_score = combined_macro_earnings_score(
                    sector,
                    ticker,
                    prices.iloc[:i],
                    market_prices.iloc[:i],
                    fred_factor_scores,
                )

                sector_etf = MARKET_TICKERS.get(sector)
                rs_market = ret_30d - pct_return(market_prices.iloc[:i], "SPY")
                rs_sector = ret_30d - pct_return(market_prices.iloc[:i], sector_etf)

                recent_vol = realised_vol(current.iloc[-15:])
                older_vol = realised_vol(current.iloc[:15])
                vol_comp = older_vol - recent_vol

                implied_score = (
                    40 * rs_sector
                    + 25 * rs_market
                    + 20 * trend
                    + 15 * vol_comp
                    + 10 * macro_total
                )

                exit_reason = None

                if pval <= p_threshold:
                    exit_reason = "Normality broken"
                elif trend <= trend_exit_threshold:
                    exit_reason = "Positive trend broken"
                elif implied_score <= implied_score_exit_threshold:
                    exit_reason = "Implied revision score materially negative"

                if exit_reason and holding_days.get(ticker, 0) >= min_hold_days:
                    trade_log.append({
                        "Date": signal_date,
                        "Ticker": ticker,
                        "Security": name_lookup.get(ticker, ""),
                        "Action": "EXIT",
                        "Reason": exit_reason,
                        "P-value": pval,
                        "Trend": trend,
                        "Implied revision score": implied_score,
                        "FRED macro score": fred_score,
                        "Market macro score": market_score,
                        "Combined macro score": macro_total,
                    })
                    del positions[ticker]
                    holding_days.pop(ticker, None)
                    exits_today += 1

            new_longs = []
            open_slots = max(max_longs - len(positions), 0)

            if open_slots > 0:
                for ticker in prices.columns:
                    if ticker in positions:
                        continue

                    prior = prices[ticker].iloc[i - lookback - 1:i - 1].dropna()
                    current = prices[ticker].iloc[i - lookback:i].dropna()

                    if len(prior) < lookback or len(current) < lookback:
                        continue

                    prior_p = normality_pvalue(prior, lookback)
                    current_p = normality_pvalue(current, lookback)

                    if np.isnan(prior_p) or np.isnan(current_p):
                        continue

                    if not (
                        prior_p <= p_threshold
                        and current_p > p_threshold
                        and current_p > prior_p
                    ):
                        continue

                    trend = trend_score(current)
                    ret_30d = current.iloc[-1] / current.iloc[0] - 1

                    sector = sector_lookup.get(ticker, None)
                    macro_total, fred_score, market_score = combined_macro_earnings_score(
                        sector,
                        ticker,
                        prices.iloc[:i],
                        market_prices.iloc[:i],
                        fred_factor_scores,
                    )

                    sector_etf = MARKET_TICKERS.get(sector)
                    rs_market = ret_30d - pct_return(market_prices.iloc[:i], "SPY")
                    rs_sector = ret_30d - pct_return(market_prices.iloc[:i], sector_etf)

                    recent_vol = realised_vol(current.iloc[-15:])
                    older_vol = realised_vol(current.iloc[:15])
                    vol_comp = older_vol - recent_vol

                    implied_score = (
                        40 * rs_sector
                        + 25 * rs_market
                        + 20 * trend
                        + 15 * vol_comp
                        + 10 * macro_total
                    )

                    if trend > 0 and ret_30d > 0 and implied_score > 0:
                        new_longs.append({
                            "Ticker": ticker,
                            "Rank score": implied_score,
                            "P-value": current_p,
                            "Trend": trend,
                            "Implied revision score": implied_score,
                            "FRED macro score": fred_score,
                            "Market macro score": market_score,
                            "Combined macro score": macro_total,
                        })

            if new_longs:
                new_longs = (
                    pd.DataFrame(new_longs)
                    .sort_values("Rank score", ascending=False)
                    .head(open_slots)
                )

                for _, row in new_longs.iterrows():
                    ticker = row["Ticker"]
                    positions[ticker] = "LONG"
                    holding_days[ticker] = 0
                    entries_today += 1

                    trade_log.append({
                        "Date": signal_date,
                        "Ticker": ticker,
                        "Security": name_lookup.get(ticker, ""),
                        "Action": "ENTER",
                        "Reason": "Normalised with positive trend and upward revision pressure",
                        "P-value": row["P-value"],
                        "Trend": row["Trend"],
                        "Implied revision score": row["Implied revision score"],
                        "FRED macro score": row["FRED macro score"],
                        "Market macro score": row["Market macro score"],
                        "Combined macro score": row["Combined macro score"],
                    })

        next_returns = returns.loc[trade_date]
        long_tickers = list(positions.keys())
        long_return = next_returns[long_tickers].mean() if long_tickers else 0.0

        market_slice = market_prices.iloc[:i]
        hedge = hedge_signal_at(
            market_slice,
            fred_factor_scores,
            sentiment,
            signal_date,
            hedge_size,
            nfci_loose_z,
            skew_low_z,
            momentum_euphoria_z,
            vix_low_z,
            require_market_crack,
        )

        current_hedge_size = hedge["hedge_size"]

        spy_ret = spy_returns.loc[trade_date] if trade_date in spy_returns.index else 0.0
        hedge_return = -current_hedge_size * spy_ret

        portfolio_return = long_return + hedge_return

        stock_weight = 1 / max(max_longs, 1)
        stock_trading_cost = (entries_today + exits_today) * stock_weight * stock_cost_per_side

        hedge_turnover = abs(current_hedge_size - previous_hedge_size)
        hedge_trading_cost = hedge_turnover * hedge_cost_per_side
        previous_hedge_size = current_hedge_size

        turnover_cost = stock_trading_cost + hedge_trading_cost
        portfolio_return -= turnover_cost

        results.append({
            "Date": trade_date,
            "Return": portfolio_return,
            "Trading cost": turnover_cost,
            "Stock trading cost": stock_trading_cost,
            "Hedge trading cost": hedge_trading_cost,
            "Long book return": long_return,
            "Hedge return": hedge_return,
            "SPY return": spy_ret,
            "Hedge active": hedge["hedge_on"],
            "Hedge size": current_hedge_size,
            "Loose conditions z": hedge["loose_conditions_z"],
            "SKEW z": hedge["skew_z"],
            "SPY 3m momentum z": hedge["spy_3m_momentum_z"],
            "VIX z": hedge["vix_z"],
            "SPY crack flag": hedge["spy_crack_flag"],
            "Low SKEW flag": hedge["low_skew_flag"],
            "Low VIX flag": hedge["complacent_vix_flag"],
            "Momentum euphoria flag": hedge["euphoric_momentum_flag"],
            "Rebalance day": is_rebalance_day,
            "Entries": entries_today,
            "Exits": exits_today,
            "Number longs": len(long_tickers),
            "Longs": ", ".join([f"{t} ({name_lookup.get(t, '')})" for t in long_tickers]),
        })

    return pd.DataFrame(results), pd.DataFrame(trade_log)


bt, trades = run_long_with_tactical_hedge_backtest(
    prices,
    sp500,
    market_prices,
    sentiment,
    fred_factor_scores,
    LOOKBACK,
    P_THRESHOLD,
    MAX_LONGS,
    MIN_HOLD_DAYS,
    REBALANCE_WEEKDAY,
    TREND_EXIT_THRESHOLD,
    IMPLIED_SCORE_EXIT_THRESHOLD,
    STOCK_COST_PER_SIDE,
    HEDGE_COST_PER_SIDE,
    HEDGE_SIZE,
    NFCI_LOOSE_Z,
    SKEW_LOW_Z,
    MOMENTUM_EUPHORIA_Z,
    VIX_LOW_Z,
    REQUIRE_MARKET_CRACK,
)

bt = show_backtest("Long normalisation strategy + tactical SPY hedge", bt)

st.markdown("### Current portfolio and today's recommended actions")

if bt is None or bt.empty:
    st.write("No current portfolio available.")
else:
    latest_row = bt.iloc[-1]
    current_longs = latest_row["Longs"]
    hedge_active = latest_row["Hedge active"]
    hedge_size_now = latest_row["Hedge size"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current long positions", int(latest_row["Number longs"]))
    c2.metric("SPY hedge active", "Yes" if hedge_active else "No")
    c3.metric("SPY hedge size", f"{hedge_size_now:.0%}")

    st.markdown("#### Current holdings")

    holdings_rows = []

    if isinstance(current_longs, str) and current_longs.strip() != "":
        for holding in current_longs.split(", "):
            holdings_rows.append({
                "Position": "LONG",
                "Holding": holding,
                "Size": "Equal-weight long book",
            })

    if hedge_active:
        holdings_rows.append({
            "Position": "SHORT",
            "Holding": "SPY",
            "Size": f"{hedge_size_now:.0%} hedge",
        })

    if holdings_rows:
        st.dataframe(pd.DataFrame(holdings_rows), use_container_width=True)
    else:
        st.write("No active positions.")

    st.markdown("#### Today's recommended trades")

    action_rows = []

    if not trades.empty:
        latest_trade_date = trades["Date"].max()
        todays_trades = trades[trades["Date"] == latest_trade_date]

        for _, row in todays_trades.iterrows():
            if row["Action"] == "ENTER":
                action_rows.append({
                    "Action": "BUY",
                    "Ticker": row["Ticker"],
                    "Security": row.get("Security", ""),
                    "Reason": row.get("Reason", "New long signal"),
                    "Status": "Added to portfolio",
                })

            elif row["Action"] == "EXIT":
                action_rows.append({
                    "Action": "SELL",
                    "Ticker": row["Ticker"],
                    "Security": row.get("Security", ""),
                    "Reason": row.get("Reason", "Exit signal"),
                    "Status": "Removed from portfolio",
                })

    if len(bt) >= 2:
        previous_row = bt.iloc[-2]
        previous_hedge = bool(previous_row["Hedge active"])
        current_hedge = bool(latest_row["Hedge active"])

        if current_hedge and not previous_hedge:
            action_rows.append({
                "Action": "ADD HEDGE",
                "Ticker": "SPY",
                "Security": "SPDR S&P 500 ETF Trust",
                "Reason": "Loose conditions + complacency/euphoria + SPY crack filter",
                "Status": "Added to portfolio",
            })

        elif previous_hedge and not current_hedge:
            action_rows.append({
                "Action": "REMOVE HEDGE",
                "Ticker": "SPY",
                "Security": "SPDR S&P 500 ETF Trust",
                "Reason": "Hedge trigger no longer active",
                "Status": "Removed from portfolio",
            })

    if action_rows:
        st.dataframe(pd.DataFrame(action_rows), use_container_width=True)
    else:
        st.success("No new trades today. Maintain current portfolio.")

st.markdown("### Portfolio history")

if bt is not None and not bt.empty:
    cols = [
        "Date",
        "Return",
        "Trading cost",
        "Stock trading cost",
        "Hedge trading cost",
        "Long book return",
        "Hedge return",
        "SPY return",
        "Hedge active",
        "Hedge size",
        "Loose conditions z",
        "SKEW z",
        "SPY 3m momentum z",
        "VIX z",
        "SPY crack flag",
        "Low SKEW flag",
        "Low VIX flag",
        "Momentum euphoria flag",
        "Rebalance day",
        "Entries",
        "Exits",
        "Number longs",
        "Longs",
    ]

    st.dataframe(bt[cols].tail(100), use_container_width=True)

    csv = bt[cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download portfolio history as CSV",
        data=csv,
        file_name="long_with_tactical_hedge_history.csv",
        mime="text/csv",
    )

st.markdown("### Trade log")

if trades.empty:
    st.write("No trades.")
else:
    st.dataframe(trades.tail(100), use_container_width=True)

    csv_trades = trades.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download trade log as CSV",
        data=csv_trades,
        file_name="long_with_tactical_hedge_trade_log.csv",
        mime="text/csv",
    )