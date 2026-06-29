import numpy as np
import pandas as pd
import datetime

# Attempt to import yfinance. If not present, we will fallback to mock data dynamically.
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Hardcoded high-quality mock data for fallbacks (useful for offline demo or dependency issues)
MOCK_DATA = {
    "AAPL": {"price": 185.50, "volatility": 0.22, "name": "Apple Inc."},
    "TSLA": {"price": 178.20, "volatility": 0.48, "name": "Tesla, Inc."},
    "MSFT": {"price": 415.60, "volatility": 0.19, "name": "Microsoft Corporation"},
    "GOOG": {"price": 150.10, "volatility": 0.24, "name": "Alphabet Inc."},
    "NVDA": {"price": 875.12, "volatility": 0.42, "name": "NVIDIA Corporation"},
}

def get_stock_data(ticker: str) -> dict:
    """
    Fetch stock metadata, current price, and compute historical volatility.
    Falls back to mock data if yfinance fails or is not installed.
    """
    ticker = ticker.upper().strip()
    
    if not YFINANCE_AVAILABLE:
        # Fallback to mock data
        return get_mock_stock_data(ticker, "yfinance is not installed (running in offline mode)")
        
    try:
        # Fetch stock object
        stock = yf.Ticker(ticker)
        
        # Get historical data for the last 1 year (252 trading days)
        hist = stock.history(period="1y")
        
        if hist.empty or len(hist) < 10:
            return get_mock_stock_data(ticker, f"No historical data returned for ticker {ticker}")
            
        # Get current price (most recent close)
        current_price = float(hist["Close"].iloc[-1])
        
        # Calculate daily log returns: ln(P_t / P_{t-1})
        close_prices = hist["Close"]
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Calculate standard deviation of daily log returns
        daily_std = log_returns.std()
        
        # Annualize volatility: daily_std * sqrt(252)
        annualized_vol = float(daily_std * np.sqrt(252))
        
        # Get company name
        info = stock.info
        company_name = info.get("longName", f"{ticker} Corporation")
        
        # Prepare historical price series for the chart (last 30 days for simplicity)
        recent_hist = hist.tail(30)
        history_points = []
        for index, row in recent_hist.iterrows():
            history_points.append({
                "date": index.strftime("%Y-%m-%d"),
                "close": float(row["Close"])
            })
            
        return {
            "ticker": ticker,
            "company_name": company_name,
            "current_price": current_price,
            "historical_volatility": annualized_vol,
            "history": history_points,
            "source": "Yahoo Finance (Live)"
        }
        
    except Exception as e:
        return get_mock_stock_data(ticker, f"Error fetching from yfinance: {str(e)}")

def get_mock_stock_data(ticker: str, warning: str = "") -> dict:
    """
    Generates realistic mock data for standard tickers or random walks for custom tickers.
    """
    # If standard ticker is matched
    if ticker in MOCK_DATA:
        data = MOCK_DATA[ticker]
        price = data["price"]
        vol = data["volatility"]
        name = data["name"]
    else:
        # Generate random yet stable values for unknown tickers
        price = 100.0 + (hash(ticker) % 150)
        vol = 0.15 + ((hash(ticker) % 40) / 100.0)
        name = f"{ticker} Inc."
        
    # Generate 30 days of mock prices (using a simple random walk)
    np.random.seed(abs(hash(ticker)) % 1000)
    history_points = []
    current_date = datetime.date.today() - datetime.timedelta(days=30)
    
    mock_price = price * 0.95  # start slightly lower
    for i in range(30):
        # random step
        step = mock_price * (vol / np.sqrt(252)) * np.random.standard_normal()
        mock_price += step
        history_points.append({
            "date": (current_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
            "close": float(mock_price)
        })
        
    # Make sure last point equals the final price
    history_points[-1]["close"] = price

    return {
        "ticker": ticker,
        "company_name": name,
        "current_price": price,
        "historical_volatility": vol,
        "history": history_points,
        "source": f"Mock Data (Offline) - {warning}" if warning else "Mock Data (Offline)"
    }
