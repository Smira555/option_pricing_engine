import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Ensure current directory is in search path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.black_scholes import black_scholes_price, black_scholes_greeks
from engine.monte_carlo import monte_carlo_price
from engine.implied_vol import calculate_implied_volatility
from engine.data_fetcher import get_stock_data

app = FastAPI(
    title="Options Pricing Engine API",
    description="Backend API for option pricing models, Monte Carlo path simulations, and implied volatility calculations.",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Request Models
class PricingRequest(BaseModel):
    S: float = Field(..., description="Underlying stock price", gt=0)
    K: float = Field(..., description="Strike price", gt=0)
    T: float = Field(..., description="Time to maturity in years", ge=0)
    r: float = Field(..., description="Risk-free rate (annualized, e.g. 0.05)", ge=-0.1, le=0.5)
    sigma: float = Field(..., description="Volatility (annualized, e.g. 0.20)", ge=0.0, le=3.0)
    option_type: str = Field("call", description="'call' or 'put'")
    num_simulations: int = Field(50000, description="Monte Carlo simulation iterations", ge=100, le=200000)
    use_antithetic: bool = Field(True, description="Enable Antithetic Variates variance reduction")
    market_price: Optional[float] = Field(None, description="Observed market price to calculate Implied Volatility", ge=0.0)

@app.post("/api/price", response_model=Dict[str, Any])
def price_option(request: PricingRequest):
    try:
        opt_type = request.option_type.lower()
        if opt_type not in ["call", "put"]:
            raise HTTPException(status_code=400, detail="option_type must be 'call' or 'put'")

        # 1. Calculate Analytical Black-Scholes Price
        bs_price = black_scholes_price(
            S=request.S, K=request.K, T=request.T, r=request.r, sigma=request.sigma, option_type=opt_type
        )

        # 2. Calculate Analytical Greeks
        greeks = black_scholes_greeks(
            S=request.S, K=request.K, T=request.T, r=request.r, sigma=request.sigma, option_type=opt_type
        )

        # 3. Calculate Monte Carlo Simulation Price
        mc_results = monte_carlo_price(
            S=request.S, K=request.K, T=request.T, r=request.r, sigma=request.sigma,
            option_type=opt_type, num_simulations=request.num_simulations, use_antithetic=request.use_antithetic
        )

        # 4. Calculate Implied Volatility if market price is provided
        implied_vol = None
        if request.market_price is not None:
            implied_vol = calculate_implied_volatility(
                S=request.S, K=request.K, T=request.T, r=request.r,
                market_price=request.market_price, option_type=opt_type
            )

        return {
            "analytical": {
                "price": bs_price,
                "greeks": greeks
            },
            "monte_carlo": {
                "price": mc_results["price"],
                "std_err": mc_results["std_err"],
                "ci_lower": mc_results["ci_lower"],
                "ci_upper": mc_results["ci_upper"],
                "paths": mc_results["paths"],
                "time_grid": mc_results["time_grid"],
                "convergence": mc_results["convergence"]
            },
            "implied_volatility": implied_vol
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pricing error: {str(e)}")

@app.get("/api/stock")
def get_stock_market_data(ticker: str):
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker parameter is required")
    data = get_stock_data(ticker)
    return data

# Mount static frontend files. We expect the frontend folder to be adjacent to backend folder.
# We will create frontend files there. If running locally, you can access the page directly.
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    # If frontend dir does not exist, log it but don't fail starting the API
    print(f"Warning: Frontend directory not found at {frontend_path}. API endpoints are active, but UI cannot be served directly.")
