"""
Real market data fetcher using yfinance.

What this fetches and why
-------------------------
S0  (spot price)       : latest closing price from Yahoo Finance.
sigma (historical vol) : annualized standard deviation of daily log-returns
                         over the past `vol_lookback` trading days.
                         This is "realized volatility" -- the simplest
                         estimate of how much the stock has actually wiggled.
                         In practice traders often prefer implied volatility
                         (backed out from option market prices), but that
                         requires an options data feed. Historical vol is
                         freely available and a reasonable first estimate.
r   (risk-free rate)   : approximated from the 13-week US Treasury bill yield
                         via Yahoo Finance ticker ^IRX (annualized, divided
                         by 100). Falls back to 0.05 if fetch fails.
"""

import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def fetch_market_data(ticker: str, vol_lookback: int = 252) -> dict:
    """
    Fetch real market data for a given stock ticker.

    Parameters
    ----------
    ticker       : stock symbol, e.g. 'AAPL', 'MSFT', 'TSLA'
    vol_lookback : number of trading days used to estimate historical vol
                   (252 = one calendar year of trading days)

    Returns
    -------
    dict with keys:
        ticker, name, sector, S0, sigma, r,
        price_history (DataFrame), returns (array)
    """
    stock = yf.Ticker(ticker.upper())

    # --- price history ---
    hist = stock.history(period=f"{max(vol_lookback + 50, 300)}d")
    if hist.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. "
                         "Check the symbol and try again.")

    closes = hist["Close"].dropna()
    S0 = float(closes.iloc[-1])

    # --- historical volatility ---
    log_returns = np.log(closes / closes.shift(1)).dropna()
    recent = log_returns.iloc[-vol_lookback:]
    sigma = float(recent.std() * np.sqrt(252))   # annualize daily vol

    # --- risk-free rate from 13-week T-bill ---
    try:
        tbill = yf.Ticker("^IRX")
        tbill_hist = tbill.history(period="5d")
        r = float(tbill_hist["Close"].iloc[-1]) / 100.0
    except Exception:
        r = 0.05   # fallback if rate fetch fails

    # --- company info (best-effort, some tickers may not have it) ---
    try:
        info = stock.info
        name   = info.get("longName", ticker.upper())
        sector = info.get("sector", "—")
        currency = info.get("currency", "USD")
    except Exception:
        name, sector, currency = ticker.upper(), "—", "USD"

    return {
        "ticker":        ticker.upper(),
        "name":          name,
        "sector":        sector,
        "currency":      currency,
        "S0":            S0,
        "sigma":         sigma,
        "r":             r,
        "price_history": closes,
        "log_returns":   np.array(recent),
    }


def get_treasury_rate() -> float:
    """Standalone helper to fetch the current 13-week T-bill rate."""
    try:
        tbill = yf.Ticker("^IRX")
        hist = tbill.history(period="5d")
        return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        return 0.05
