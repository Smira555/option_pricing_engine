"""
Cross-validation tests: every numerical method should agree with the
closed-form Black-Scholes price within its own expected error tolerance.

Run with:  python -m pytest tests/test_pricers.py -v
or simply: python tests/test_pricers.py
"""

import sys
import os
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricers.black_scholes import bs_call_price, bs_put_price, bs_greeks
from pricers.pde_solver import bs_pde_call_price
from pricers.monte_carlo import mc_call_price, mc_call_price_euler
from pricers.fft_pricer import fft_price_at_strike
from utils.characteristic_functions import bs_char_func


CASES = [
    dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.20),   # ATM
    dict(S0=100, K=120, T=1.0, r=0.05, sigma=0.20),   # OTM
    dict(S0=100, K=80, T=1.0, r=0.05, sigma=0.20),    # ITM
    dict(S0=100, K=100, T=0.25, r=0.03, sigma=0.35),  # short-dated, high vol
]


def test_pde_matches_closed_form():
    for c in CASES:
        bs = bs_call_price(**c)
        pde = bs_pde_call_price(**c, M=400, N=2000)
        assert abs(pde - bs) < 0.05, f"PDE mismatch {c}: {pde} vs {bs}"


def test_fft_matches_closed_form():
    for c in CASES:
        bs = bs_call_price(**c)
        cf = partial(bs_char_func, S0=c["S0"], T=c["T"], r=c["r"], sigma=c["sigma"])
        fft = fft_price_at_strike(c["K"], cf, c["T"], c["r"])
        assert abs(fft - bs) < 0.01, f"FFT mismatch {c}: {fft} vs {bs}"


def test_monte_carlo_matches_closed_form():
    for c in CASES:
        bs = bs_call_price(**c)
        mc, se, (lo, hi) = mc_call_price(**c, n_paths=300_000, seed=7)
        assert lo - 4 * se <= bs <= hi + 4 * se, f"MC mismatch {c}: {mc}+-{se} vs {bs}"


def test_monte_carlo_euler_matches_closed_form():
    for c in CASES:
        bs = bs_call_price(**c)
        mc, se = mc_call_price_euler(**c, n_paths=40_000, n_steps=100, seed=7)
        assert abs(mc - bs) < 8 * se, f"Euler MC mismatch {c}: {mc}+-{se} vs {bs}"


def test_put_call_parity():
    for c in CASES:
        call = bs_call_price(**c)
        put = bs_put_price(**c)
        import numpy as np
        lhs = call - put
        rhs = c["S0"] - c["K"] * np.exp(-c["r"] * c["T"])
        assert abs(lhs - rhs) < 1e-8, f"Put-call parity violated: {lhs} vs {rhs}"


def test_greeks_sane():
    g = bs_greeks(S0=100, K=100, T=1.0, r=0.05, sigma=0.20)
    assert 0.0 < g["delta"] < 1.0
    assert g["gamma"] > 0
    assert g["vega"] > 0


def test_zero_vol_limit():
    # As sigma -> 0, the call price collapses to the discounted intrinsic value
    bs = bs_call_price(S0=100, K=90, T=1.0, r=0.05, sigma=1e-6)
    intrinsic = 100 - 90 * 2.718281828 ** (-0.05)
    assert abs(bs - intrinsic) < 1e-3


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
