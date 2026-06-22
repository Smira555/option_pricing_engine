"""
Monte Carlo European call pricer, grounded in stochastic calculus.

JUSTIFICATION
-------------
By the Feynman-Kac theorem, the Black-Scholes PDE solution equals a
risk-neutral expectation:
    V(S0, 0) = exp(-rT) * E_Q[ max(S_T - K, 0) ]
where S_T evolves under dS = r S dt + sigma S dW (risk-neutral GBM).

Applying Ito's lemma to f(S) = ln(S) gives d(ln S) = (r - 0.5*sigma^2)dt +
sigma dW, which integrates EXACTLY (GBM is one of the few SDEs with a
closed-form path solution):
    S_T = S0 * exp( (r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z ),   Z ~ N(0,1)

So we can sample S_T directly with zero discretization error and just
average the discounted payoff (a direct Monte Carlo estimate of the
Feynman-Kac expectation). This is the "exact simulation" pricer.

We also include an Euler-Maruyama path simulator, which discretizes the
SDE step by step instead of using the closed-form solution. For plain
GBM this is strictly worse (it has O(sqrt(dt)) discretization bias) and
exists here mainly to (a) demonstrate the general-purpose Monte Carlo
machinery used in stochastic calculus when no exact solution exists
(e.g. local vol, Heston), and (b) let you empirically measure that bias
against the exact method in the same script.
"""

import numpy as np


def mc_call_price(S0: float, K: float, T: float, r: float, sigma: float,
                   n_paths: int = 200_000, antithetic: bool = True, seed: int | None = None):
    """
    Exact-simulation Monte Carlo price via the closed-form GBM solution.
    Returns (price, standard_error, 95%_confidence_interval).

    Antithetic variates (using both Z and -Z) are used as a variance
    reduction technique: it costs nothing extra to compute and removes a
    meaningful chunk of estimator variance because the payoff is a
    monotonic function of Z.
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_paths)

    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S0 * np.exp(drift + diffusion)

    payoff = np.maximum(S_T - K, 0.0)
    discounted = np.exp(-r * T) * payoff

    price = float(discounted.mean())
    se = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))
    ci95 = (price - 1.96 * se, price + 1.96 * se)
    return price, se, ci95


def mc_call_price_euler(S0: float, K: float, T: float, r: float, sigma: float,
                         n_paths: int = 50_000, n_steps: int = 252, seed: int | None = None):
    """
    Euler-Maruyama discretization of dS = r*S*dt + sigma*S*dW, simulated
    step by step. Included for comparison against the exact-simulation
    method (mc_call_price) to illustrate discretization bias, and because
    this is the technique that *does* generalize to SDEs without a known
    closed-form solution.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, S0, dtype=float)

    for _ in range(n_steps):
        Z = rng.standard_normal(n_paths)
        S = S + r * S * dt + sigma * S * np.sqrt(dt) * Z
        S = np.maximum(S, 1e-8)  # guard against (rare) negative steps

    payoff = np.maximum(S - K, 0.0)
    discounted = np.exp(-r * T) * payoff

    price = float(discounted.mean())
    se = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price, se
