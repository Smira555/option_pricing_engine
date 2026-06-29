from backend.engine.black_scholes import black_scholes_price, black_scholes_greeks

def calculate_implied_volatility(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    market_price: float, 
    option_type: str = "call", 
    max_iterations: int = 100, 
    tolerance: float = 1e-6
) -> float:
    """
    Calculate the implied volatility of a European option using the Newton-Raphson method.
    Falls back to Bisection method if Newton-Raphson diverges or encounters flat Vega.
    """
    # Quick checks
    intrinsic_value = max(S - K, 0.0) if option_type.lower() == "call" else max(K - S, 0.0)
    
    # Market price must be at least the discounted intrinsic value, and less than the stock price (for calls)
    if market_price <= intrinsic_value:
        return 0.0
    
    # Try Newton-Raphson first
    sigma = 0.20  # Initial guess (20% volatility)
    
    for i in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
        vega = greeks["vega"]
        
        # Avoid division by zero if Vega is too small
        if abs(vega) < 1e-4:
            break
            
        diff = price - market_price
        if abs(diff) < tolerance:
            return float(sigma)
            
        # Update guess
        sigma_new = sigma - diff / vega
        
        # If the new sigma is negative or extremely large, fall back to bisection
        if sigma_new <= 0 or sigma_new > 5.0:
            break
            
        sigma = sigma_new

    # Fallback: Bisection Method (extremely robust, always converges if solution exists)
    low = 0.0001
    high = 5.0
    
    for _ in range(max_iterations):
        mid = (low + high) / 2
        price = black_scholes_price(S, K, T, r, mid, option_type)
        
        if abs(price - market_price) < tolerance:
            return float(mid)
            
        if price > market_price:
            high = mid
        else:
            low = mid
            
    return float((low + high) / 2)
