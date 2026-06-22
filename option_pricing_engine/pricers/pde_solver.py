"""
Black-Scholes PDE solver via Crank-Nicolson finite differences.

JUSTIFICATION
-------------
The closed-form solution only exists for plain European payoffs under
constant-coefficient GBM. The moment you need American exercise, barrier
features, local volatility sigma(S,t), or any payoff without a known
integral, you fall back to solving the PDE numerically. This module
solves the same Black-Scholes PDE as black_scholes.py, but with a finite
difference scheme, so it generalizes to those cases while still being
checkable against the closed form here.

We transform to log-price x = ln(S). With U(x, t) = V(e^x, t), the PDE
    dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0
becomes a CONSTANT-COEFFICIENT diffusion-advection-reaction equation:
    dU/dt + 0.5*sigma^2*d2U/dx2 + (r - 0.5*sigma^2)*dU/dx - r*U = 0

which is much better behaved on a uniform finite-difference grid than the
original S-space PDE (whose diffusion coefficient grows like S^2).

We then substitute tau = T - t (time-to-maturity) to turn it into a
forward-in-time problem starting from the known terminal payoff, and
discretize with Crank-Nicolson: average of the explicit and implicit
(backward Euler) finite-difference operators. Crank-Nicolson is chosen
because it is second-order accurate in both time and space and
unconditionally stable, unlike pure explicit differencing.
"""

import numpy as np
from scipy.linalg import solve_banded


def bs_pde_call_price(S0: float, K: float, T: float, r: float, sigma: float,
                       M: int = 400, N: int = 2000, return_grid: bool = False):
    """
    Price a European call by solving the Black-Scholes PDE with
    Crank-Nicolson finite differences on a log-price grid.

    Parameters
    ----------
    M : number of spatial (log-price) grid steps
    N : number of time steps
    return_grid : if True, also return the (S, price) grid at t=0

    Boundary conditions (European call):
      S -> 0      : V = 0
      S -> S_max  : V = S - K*exp(-r*tau)   (deep ITM, exercise ~certain)
    """
    # Grid centered on log(K), wide enough to keep boundary effects away
    # from the region of interest (a standard rule of thumb is ~5 standard
    # deviations of the terminal log-return distribution, plus a margin).
    spread = 5 * sigma * np.sqrt(T) + 1.0
    x_min = np.log(K) - spread
    x_max = np.log(K) + spread

    dx = (x_max - x_min) / M
    dtau = T / N
    x = np.linspace(x_min, x_max, M + 1)
    S = np.exp(x)

    U = np.maximum(S - K, 0.0)  # terminal payoff at tau = 0  (t = T)

    A = 0.5 * sigma ** 2 / dx ** 2
    B = (r - 0.5 * sigma ** 2) / (2 * dx)
    lo = A - B    # coefficient of U_{j-1}
    di = -2 * A - r  # coefficient of U_j
    up = A + B    # coefficient of U_{j+1}

    n_int = M - 1  # number of interior unknowns

    # Implicit (left-hand) tridiagonal system, constant across time steps
    main_imp = (1 - 0.5 * dtau * di) * np.ones(n_int)
    sub_imp = -0.5 * dtau * lo
    sup_imp = -0.5 * dtau * up

    ab = np.zeros((3, n_int))  # banded storage for scipy.linalg.solve_banded
    ab[0, 1:] = sup_imp
    ab[1, :] = main_imp
    ab[2, :-1] = sub_imp

    S_max = S[-1]

    for n in range(N):
        tau_old = n * dtau
        tau_new = (n + 1) * dtau

        bound_low_old, bound_high_old = 0.0, S_max - K * np.exp(-r * tau_old)
        bound_low_new, bound_high_new = 0.0, S_max - K * np.exp(-r * tau_new)

        # Explicit (right-hand) side, built from the known solution at tau_old
        rhs = ((1 + 0.5 * dtau * di) * U[1:M]
               + 0.5 * dtau * lo * np.r_[bound_low_old, U[1:M - 1]]
               + 0.5 * dtau * up * np.r_[U[2:M], bound_high_old])

        # Move the new (unknown side) boundary terms onto the RHS
        rhs[0] += 0.5 * dtau * lo * bound_low_new
        rhs[-1] += 0.5 * dtau * up * bound_high_new

        U_int_new = solve_banded((1, 1), ab, rhs)
        U = np.r_[bound_low_new, U_int_new, bound_high_new]

    price = float(np.interp(np.log(S0), x, U))
    if return_grid:
        return price, S, U
    return price
