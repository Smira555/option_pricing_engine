"""
Black-Scholes closed-form European call pricer.

JUSTIFICATION
-------------
Under the risk-neutral measure, the underlying follows geometric Brownian
motion (GBM): dS = r S dt + sigma S dW. Applying Ito's lemma to ln(S) and
then the Feynman-Kac theorem, the no-arbitrage price of any European claim
V(S, t) solves the Black-Scholes PDE:

    dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0

For a call payoff max(S_T - K, 0), this PDE has an exact analytical
solution (Black & Scholes, 1973). It is included here as the reference
("ground truth") against which the PDE, Monte Carlo, and FFT engines in
this project are validated, and because it is the fastest possible method
whenever its assumptions (constant r, constant sigma, no dividends,
lognormal terminal distribution) actually hold.
"""

import numpy as np
from scipy.stats import norm


def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Closed-form European call price."""
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Closed-form European put price (via put-call parity check available in tests)."""
    if T <= 0:
        return max(K - S0, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def bs_greeks(S0: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Analytical Greeks for the European call, derived by differentiating
    the closed-form price with respect to each input."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
