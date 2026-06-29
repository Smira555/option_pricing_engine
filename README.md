# Options Pricing & Analytics Engine

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive quantitative finance engine for pricing European options and analyzing their risk profiles. It implements analytical closed-form equations (Black-Scholes-Merton), stochastic path simulations (Monte Carlo with variance reduction), and numerical root-finding (Newton-Raphson for Implied Volatility).

The engine is backed by a modular Python + NumPy backend, and visualizes calculations through an interactive Streamlit dashboard featuring high-performance Plotly charts.

---

## Key Features

1.  **Analytical Black-Scholes Model**: 
    *   Instant closed-form Call & Put pricing.
    *   Analytical calculation of all primary Greeks: Delta ($\Delta$), Gamma ($\Gamma$), Vega ($\nu$), Theta ($\Theta$), and Rho ($\rho$).
2.  **Stochastic Monte Carlo Pricing Engine**:
    *   Simulates asset trajectories using Geometric Brownian Motion (GBM).
    *   Calculates 95% confidence intervals and standard errors.
    *   **Antithetic Variates** variance reduction technique: Reduces path variance by roughly 50% without generating additional random variables.
3.  **Newton-Raphson Implied Volatility Solver**:
    *   Numerically solves for implied volatility ($\sigma$) given an option's market price.
    *   Uses Vega as the gradient for rapid quadratic convergence, falling back to a robust Bisection method if volatility bounds are breached.
4.  **Live Market Data Fetcher**:
    *   Fetches current underlying prices and historical daily data (via Yahoo Finance).
    *   Automatically computes annualized historical volatility ($\sigma = \text{std}(\text{returns}) \times \sqrt{252}$).
5.  **Interactive Visualizations (Plotly)**:
    *   *Path Simulation Chart*: Hover over individual simulated Geometric Brownian Motion price paths.
    *   *MC Convergence Chart*: Zoom into running estimate convergence against the analytical price baseline.
    *   *Price Sensitivity Chart*: Plot option value and intrinsic payoff across spot prices dynamically.

---

## Directory Structure

```text
options-pricing-engine/
├── backend/
│   ├── requirements.txt        # Backend dependencies (FastAPI, NumPy, SciPy, Streamlit, Plotly)
│   └── engine/
│       ├── __init__.py
│       ├── black_scholes.py    # Analytical models & greeks
│       ├── monte_carlo.py      # Path simulator & Antithetic Variates
│       ├── implied_vol.py      # Newton-Raphson solver
│       └── data_fetcher.py     # yfinance data fetching & volatility
├── app.py                      # Streamlit interactive dashboard
├── .gitignore
├── run.bat                     # Automation script to setup & launch
└── README.md                   # Documentation & Interview Prep Guide
```

---

## Mathematical Details

### Geometric Brownian Motion
Stock prices are modeled as a stochastic process:
$$dS_t = r S_t dt + \sigma S_t dW_t$$

Under the risk-neutral measure, the analytical solution at maturity $T$ is:
$$S_T = S_0 \exp\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z\right), \quad Z \sim N(0, 1)$$

### Antithetic Variates (Variance Reduction)
For every generated path using random draw $Z$, we generate a mirrored path using $-Z$:
*   $S_T^{(1)} = S_0 \exp\left(\text{drift} + \sigma \sqrt{T} Z\right)$
*   $S_T^{(2)} = S_0 \exp\left(\text{drift} - \sigma \sqrt{T} Z\right)$

The payoffs are averaged: $\bar{P} = \frac{Payoff(S_T^{(1)}) + Payoff(S_T^{(2)})}{2}$. Since $Z$ and $-Z$ are perfectly negatively correlated, the covariance term in the sample variance is negative, reducing the standard error of the Monte Carlo estimate:
$$\text{Var}\left(\frac{X_1 + X_2}{2}\right) = \frac{1}{4} \left(\text{Var}(X_1) + \text{Var}(X_2) + 2\text{Cov}(X_1, X_2)\right)$$

---

## Installation & Running

This project includes a convenient `run.bat` script that automates Python virtual environment configuration and starts the Streamlit server.

1.  Clone the repository.
2.  Double-click `run.bat` (or execute it in terminal):
    ```bash
    run.bat
    ```
3.  Open your browser and navigate to: [http://localhost:8501](http://localhost:8501)

---
