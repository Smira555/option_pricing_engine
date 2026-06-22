"""
Risk-neutral characteristic functions of log(S_T), used by the FFT pricer.

A characteristic function phi(u) = E[exp(i*u*ln(S_T))] fully determines the
distribution of ln(S_T) (it's the Fourier transform of the log-price
density). Several important models have a closed-form characteristic
function even when they have NO closed-form option price -- that gap is
exactly what the FFT (Carr-Madan) method in fft_pricer.py is built to
exploit.
"""

import numpy as np


def bs_char_func(u, S0: float, T: float, r: float, sigma: float):
    """
    Characteristic function of ln(S_T) under Black-Scholes GBM dynamics.
    Since ln(S_T) ~ Normal(mu, sigma^2*T) under the risk-neutral measure,
    this is just the standard Gaussian characteristic function.
    """
    mu = np.log(S0) + (r - 0.5 * sigma ** 2) * T
    return np.exp(1j * u * mu - 0.5 * sigma ** 2 * T * u ** 2)


def heston_char_func(u, S0: float, T: float, r: float, v0: float,
                      kappa: float, theta: float, sigma_v: float, rho: float):
    """
    Characteristic function of ln(S_T) under the Heston (1993) stochastic
    volatility model:
        dS = r S dt + sqrt(v) S dW1
        dv = kappa*(theta - v) dt + sigma_v*sqrt(v) dW2,   corr(dW1,dW2)=rho

    Heston has NO closed-form call price (the variance process makes the
    terminal distribution non-Gaussian with no clean integral), but Heston
    (1993) derived this closed-form characteristic function by solving a
    Riccati ODE system. This is the textbook example of why FFT pricing
    matters: it turns "no closed form" into "O(N log N) prices across all
    strikes" whenever phi(u) is known analytically.
    """
    x0 = np.log(S0)
    a = kappa * theta
    b = kappa

    d = np.sqrt((rho * sigma_v * 1j * u - b) ** 2 + (sigma_v ** 2) * (1j * u + u ** 2))
    g = (b - rho * sigma_v * 1j * u - d) / (b - rho * sigma_v * 1j * u + d)

    C = (r * 1j * u * T
         + (a / sigma_v ** 2) * ((b - rho * sigma_v * 1j * u - d) * T
                                  - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
    D = ((b - rho * sigma_v * 1j * u - d) / sigma_v ** 2) * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

    return np.exp(C + D * v0 + 1j * u * x0)
