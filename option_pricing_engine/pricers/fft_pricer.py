"""
Carr-Madan FFT European call pricer (Carr & Madan, 1999).

JUSTIFICATION
-------------
The risk-neutral call price C(K) = exp(-rT) E[max(S_T-K, 0)], viewed as a
function of log-strike k = ln(K), is NOT square-integrable (it tends to
S0, not 0, as k -> -infinity), so its Fourier transform doesn't exist in
the ordinary sense -- you can't just FFT the payoff directly.

Carr-Madan's fix: price a damped call c(k) = exp(alpha*k) * C(k) for some
alpha > 0. This decays at both ends of the log-strike axis, so it DOES
have a well-defined Fourier transform psi(u), expressible analytically in
terms of the characteristic function phi(u) of ln(S_T):

    psi(u) = exp(-rT) * phi(u - (alpha+1)i) / (alpha^2 + alpha - u^2 + i(2*alpha+1)*u)

Recovering C(k) is then a single inverse Fourier integral:
    C(k) = exp(-alpha*k)/pi * Integral[ exp(-i*u*k) * psi(u) du ]

Discretizing that integral on a grid and applying the FFT computes prices
at N strikes simultaneously in O(N log N), instead of running a separate
numerical integration (or closed-form/PDE/Monte-Carlo solve) per strike.
The real payoff of this method is that it only needs phi(u) -- so it
works identically for Black-Scholes, Heston, Merton jump-diffusion, or
any other model with a known characteristic function, even when those
models have no closed-form option price at all (see utils/characteristic_functions.py).
"""

import numpy as np


def carr_madan_fft(char_func, T: float, r: float, alpha: float = 1.5,
                    N: int = 4096, eta: float = 0.25):
    """
    Compute European call prices across a grid of strikes via the
    Carr-Madan FFT method.

    Parameters
    ----------
    char_func : callable
        u -> phi(u), the characteristic function of ln(S_T) under the
        risk-neutral measure (S0 should already be baked into char_func
        via a closure/partial -- see utils/characteristic_functions.py).
    alpha : damping factor (Carr-Madan recommend ~1.5 for typical equity
        vol levels; must satisfy alpha > 0 and E[S_T^(alpha+1)] < infinity).
    N : number of FFT points (power of 2 for FFT efficiency).
    eta : spacing of the integration grid in u-space. Smaller eta ->
        wider strike range but coarser strike spacing (eta * lambda = 2*pi/N
        ties grid fineness in u-space to grid width in log-strike space).

    Returns
    -------
    strikes, call_prices : 1D arrays, the priced strike grid and matching prices.
    """
    lambd = 2 * np.pi / (N * eta)   # log-strike grid spacing
    b = N * lambd / 2               # half-width of the log-strike grid

    u = np.arange(N) * eta
    ku = -b + lambd * np.arange(N)  # log-strike grid, centered at 0

    phi = char_func(u - (alpha + 1) * 1j)
    denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
    psi = np.exp(-r * T) * phi / denom

    # Simpson's-rule-style quadrature weights for the discretized integral
    simpson = np.ones(N)
    simpson[1:-1:2] = 4
    simpson[2:-1:2] = 2
    weights = simpson * eta / 3

    x = np.exp(1j * b * u) * psi * weights
    y = np.fft.fft(x)

    call_prices = (np.exp(-alpha * ku) / np.pi) * y.real
    strikes = np.exp(ku)
    return strikes, call_prices


def fft_price_at_strike(K: float, char_func, T: float, r: float,
                         alpha: float = 1.5, N: int = 4096, eta: float = 0.25) -> float:
    """Price a single strike K by computing the full FFT grid once and
    linearly interpolating onto K (grid spacing lambd is typically very
    fine, e.g. ~0.006 in log-strike, so interpolation error is negligible)."""
    strikes, prices = carr_madan_fft(char_func, T, r, alpha, N, eta)
    return float(np.interp(K, strikes, prices))
