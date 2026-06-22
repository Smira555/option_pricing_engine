# European Call Option Pricing Engine

A from-scratch pricing engine for European call options, implemented four
different ways: closed-form Black-Scholes, a PDE finite-difference solver,
Monte Carlo simulation, and FFT (Carr-Madan). All four are cross-validated
against each other in `tests/test_pricers.py`.

```
option_pricing_engine/
├── main.py                          # demo: prices the same option 4 ways
├── pricers/
│   ├── black_scholes.py             # closed-form price + Greeks
│   ├── pde_solver.py                # Crank-Nicolson finite differences
│   ├── monte_carlo.py               # exact GBM simulation + Euler-Maruyama
│   └── fft_pricer.py                # Carr-Madan FFT
├── utils/
│   └── characteristic_functions.py  # BS and Heston characteristic functions
├── tests/
│   └── test_pricers.py              # cross-checks all 4 methods
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py                 # prices a sample option all 4 ways
python tests/test_pricers.py   # or: pytest tests/ -v
```

## Why four methods for one option price?

A European call under plain Black-Scholes assumptions already has a
closed-form solution, so strictly you only need method #1. The other
three are included because **they are the actual tools used once you
leave that narrow setting** — American/barrier exercise, local or
stochastic volatility, no known closed-form, pricing many strikes at
once, etc. Building all four against a problem where you *know* the
right answer is the standard way to validate that a numerical engine
is correct before trusting it on a harder problem.

---

### 1. Closed-form Black-Scholes (`pricers/black_scholes.py`)

Under the risk-neutral measure the stock follows GBM, `dS = rS dt + σS dW`.
Applying **Ito's lemma** to `ln S` and the **Feynman-Kac theorem** turns
the pricing problem into the Black-Scholes PDE:

```
∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S − rV = 0
```

For a call payoff this PDE has an exact analytical solution:

```
C = S0·N(d1) − K·e^(−rT)·N(d2)
d1 = [ln(S0/K) + (r + ½σ²)T] / (σ√T),   d2 = d1 − σ√T
```

**Why use it:** it's exact and effectively free to compute (~0.3 ms here).
It's the ground truth the other three methods are checked against, and
the right tool whenever its assumptions actually hold.

### 2. Black-Scholes PDE via Crank-Nicolson finite differences (`pricers/pde_solver.py`)

The same PDE, solved numerically instead of analytically. We transform to
log-price `x = ln(S)` so the PDE has constant coefficients:

```
∂U/∂t + ½σ² ∂²U/∂x² + (r − ½σ²) ∂U/∂x − rU = 0
```

then substitute `τ = T − t` to turn it into a forward-in-time problem
starting from the known terminal payoff `max(e^x − K, 0)`. We discretize
with **Crank-Nicolson** (the average of the explicit and implicit
finite-difference operators) because it's second-order accurate in both
space and time and unconditionally stable — a pure explicit scheme would
need a much finer time grid to avoid blowing up.

**Why use it:** this is the general-purpose tool for option pricing once
you have early-exercise features, barriers, or non-constant volatility —
none of which have closed forms, but all of which are easy to bolt onto a
PDE solver (different boundary/terminal conditions, an early-exercise
constraint, etc.). It's validated here on the vanilla European case where
we can check it against #1.

### 3. Monte Carlo simulation (`pricers/monte_carlo.py`)

By Feynman-Kac, the PDE solution equals a risk-neutral expectation:

```
V(S0, 0) = e^(−rT) · E_Q[ max(S_T − K, 0) ]
```

Applying **Ito's lemma** to `ln S` shows `d(ln S) = (r − ½σ²)dt + σ dW`,
which integrates *exactly* — GBM is one of the few SDEs with a known
closed-form path solution:

```
S_T = S0 · exp[ (r − ½σ²)T + σ√T·Z ],   Z ~ N(0,1)
```

so `mc_call_price` samples `S_T` directly (zero discretization error) and
averages the discounted payoff, with antithetic variates (`Z` and `−Z`)
for variance reduction. `mc_call_price_euler` instead discretizes the SDE
step-by-step (Euler-Maruyama) — strictly worse here, but it's the
technique that generalizes to SDEs *without* a closed-form solution
(local vol, etc.), and comparing the two in `main.py` lets you see the
discretization bias directly.

**Why use it:** Monte Carlo is the only method here that scales painlessly
to high-dimensional or path-dependent payoffs (baskets, Asian options,
multi-factor models) where PDE grids become intractable.

### 4. FFT / Carr-Madan (`pricers/fft_pricer.py`)

The call price as a function of log-strike `k = ln K` isn't square-integrable
(it tends to `S0`, not `0`, as `k → −∞`), so you can't Fourier-transform it
directly. **Carr & Madan (1999)** instead price a *damped* call
`c(k) = e^(αk)·C(k)`, which decays at both ends and so has a clean Fourier
transform `ψ(u)` expressible via the characteristic function `φ(u)` of
`ln(S_T)`:

```
ψ(u) = e^(−rT) · φ(u − (α+1)i) / (α² + α − u² + i(2α+1)u)
C(k) = (e^(−αk)/π) · ∫ e^(−iuk) ψ(u) du
```

Discretizing this integral and applying the **FFT** prices an entire grid
of strikes (4096 in this project) in a single O(N log N) pass — versus
running a separate pricer per strike.

**Why use it:** the real value isn't speed on Black-Scholes (which is
already instant via #1) — it's that this method only needs `φ(u)`, the
characteristic function of the terminal log-price. Models like **Heston**
have a known closed-form `φ(u)` (via a solved Riccati ODE) but **no**
closed-form option price. The included `heston_char_func` demonstrates
this: `main.py` prices a Heston call purely from its characteristic
function, something methods #1–#3 can't do without modification.

---

## Validation

`tests/test_pricers.py` checks, across ATM/OTM/ITM and short/long-dated
cases:
- PDE price within 5 cents of closed-form
- FFT price within 1 cent of closed-form
- Monte Carlo price within its own confidence interval
- Put-call parity holds exactly for the closed-form pricer
- Greeks have sane signs/bounds
- The zero-volatility limit collapses to discounted intrinsic value

All 7 tests pass (`python tests/test_pricers.py`).

## Extending this engine

- **American options**: add an early-exercise check (`U = max(U, payoff)`)
  after each PDE time step in `pde_solver.py`, or use Longstaff-Schwartz
  regression on the Monte Carlo paths.
- **Local volatility**: replace constant `σ` in the PDE with `σ(S, t)` —
  the Crank-Nicolson scheme just needs its coefficients recomputed at
  each step.
- **Other models via FFT**: add a characteristic function (Merton
  jump-diffusion, Variance Gamma, SABR-approximations, etc.) to
  `utils/characteristic_functions.py` and pass it straight into
  `carr_madan_fft` — no changes needed elsewhere.
