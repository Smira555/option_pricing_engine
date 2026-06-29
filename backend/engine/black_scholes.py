import numpy as np
from scipy.stats import norm

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Calculate the analytical Black-Scholes price for European Call and Put options.
    S: Underling stock price
    K: Strike price
    T: Time to maturity in years
    r: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
    sigma: Volatility of the underlying stock (annualized, e.g., 0.20 for 20%)
    option_type: "call" or "put"
    """
    if T <= 0:
        if option_type.lower() == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
            
    if sigma <= 0:
        # Volatility is 0, deterministic payoff discounted
        discounted_strike = K * np.exp(-r * T)
        if option_type.lower() == "call":
            return max(S - discounted_strike, 0.0)
        else:
            return max(discounted_strike - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return float(price)

def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    """
    Calculate the analytical Greeks (Delta, Gamma, Vega, Theta, Rho) for European options.
    Returns a dictionary of greeks.
    """
    # Default outputs if maturity is reached (T <= 0)
    if T <= 0:
        is_call = option_type.lower() == "call"
        if S == K:
            delta = 0.5 if is_call else -0.5
        elif S > K:
            delta = 1.0 if is_call else 0.0
        else:
            delta = 0.0 if is_call else -1.0
            
        return {
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }

    if sigma <= 0:
        sigma = 1e-5  # avoid division by zero in calculations

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Normal PDF (d1)
    pdf_d1 = norm.pdf(d1)
    
    # Gamma and Vega are identical for both calls and puts
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        delta = norm.cdf(d1) - 1.0
        theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    # Annualized theta is typically divided by 365 or 252 to express decay per day
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),         # Annualized theta
        "theta_per_day": float(theta / 365.0), # Daily theta (decay per calendar day)
        "rho": float(rho)
    }
