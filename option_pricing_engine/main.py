"""
European Call Option Pricing Engine -- demo / entry point.

Prices the same option with four different methods:
  1. Black-Scholes closed-form     (pricers/black_scholes.py)
  2. Black-Scholes PDE, Crank-Nicolson FD (pricers/pde_solver.py)
  3. Monte Carlo (exact GBM simulation + Euler-Maruyama) (pricers/monte_carlo.py)
  4. FFT / Carr-Madan, using the BS characteristic function (pricers/fft_pricer.py)

and additionally prices a Heston-model call via FFT to show the method
working in a setting where the closed-form pricer (#1) does not apply.

Run:  python main.py
"""

import time
from functools import partial

from pricers.black_scholes import bs_call_price, bs_greeks
from pricers.pde_solver import bs_pde_call_price
from pricers.monte_carlo import mc_call_price, mc_call_price_euler
from pricers.fft_pricer import fft_price_at_strike, carr_madan_fft
from utils.characteristic_functions import bs_char_func, heston_char_func


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main():
    # ---- Option / market parameters ----
    S0 = 100.0      # spot price
    K = 100.0       # strike
    T = 1.0         # time to maturity (years)
    r = 0.05        # risk-free rate
    sigma = 0.20    # volatility

    print("=" * 64)
    print("EUROPEAN CALL OPTION PRICING ENGINE")
    print(f"S0={S0}  K={K}  T={T}y  r={r}  sigma={sigma}")
    print("=" * 64)

    # 1. Closed-form Black-Scholes -----------------------------------------
    (bs_price), t_bs = timed(bs_call_price, S0, K, T, r, sigma)
    print(f"\n[1] Closed-form Black-Scholes : {bs_price:.6f}   ({t_bs*1e3:.3f} ms)")

    greeks = bs_greeks(S0, K, T, r, sigma)
    print("    Greeks:", {k: round(float(v), 4) for k, v in greeks.items()})

    # 2. PDE (Crank-Nicolson finite differences) -----------------------------
    (pde_price), t_pde = timed(bs_pde_call_price, S0, K, T, r, sigma, 400, 2000)
    print(f"\n[2] PDE (Crank-Nicolson)       : {pde_price:.6f}   ({t_pde*1e3:.1f} ms)"
          f"   diff vs BS: {pde_price - bs_price:+.5f}")

    # 3. Monte Carlo ----------------------------------------------------------
    (mc_result), t_mc = timed(mc_call_price, S0, K, T, r, sigma, 300_000, True, 42)
    mc_price, mc_se, mc_ci = mc_result
    print(f"\n[3a] Monte Carlo (exact GBM)   : {mc_price:.6f} +/- {mc_se:.5f}"
          f"   ({t_mc*1e3:.1f} ms)   diff vs BS: {mc_price - bs_price:+.5f}")
    print(f"     95% CI: [{mc_ci[0]:.5f}, {mc_ci[1]:.5f}]")

    (mce_result), t_mce = timed(mc_call_price_euler, S0, K, T, r, sigma, 50_000, 252, 42)
    mce_price, mce_se = mce_result
    print(f"[3b] Monte Carlo (Euler-Maruyama): {mce_price:.6f} +/- {mce_se:.5f}"
          f"   ({t_mce*1e3:.1f} ms)   diff vs BS: {mce_price - bs_price:+.5f}")

    # 4. FFT / Carr-Madan -------------------------------------------------------
    cf = partial(bs_char_func, S0=S0, T=T, r=r, sigma=sigma)
    (fft_price), t_fft = timed(fft_price_at_strike, K, cf, T, r)
    print(f"\n[4] FFT (Carr-Madan)           : {fft_price:.6f}   ({t_fft*1e3:.1f} ms)"
          f"   diff vs BS: {fft_price - bs_price:+.5f}")

    # Show the FFT's real strength: pricing across an entire strike grid in one call
    t0 = time.perf_counter()
    strikes, prices = carr_madan_fft(cf, T, r)
    t_grid = time.perf_counter() - t0
    near = [(K_, P_) for K_, P_ in zip(strikes, prices) if 70 <= K_ <= 130]
    sample = near[::max(1, len(near) // 6)]
    print(f"    Full strike grid ({len(strikes)} strikes) priced in {t_grid*1e3:.1f} ms. Sample:")
    for K_, P_ in sample:
        print(f"      K={K_:7.2f}  C={P_:8.4f}")

    # ---- Bonus: Heston model via FFT (no closed-form price exists) ----------
    print("\n" + "=" * 64)
    print("BONUS: Heston stochastic-volatility model, priced via FFT")
    print("(this model has NO closed-form call price -- only its")
    print(" characteristic function is known analytically, which is")
    print(" exactly the situation Carr-Madan/FFT pricing is built for)")
    print("=" * 64)
    heston_cf = partial(heston_char_func, S0=S0, T=T, r=r,
                         v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    heston_price = fft_price_at_strike(K, heston_cf, T, r)
    print(f"Heston call price (K={K}): {heston_price:.4f}"
          f"   (vs flat-vol BS price {bs_price:.4f} -- difference reflects the vol smile)")

    print("\nDone.")


if __name__ == "__main__":
    main()
