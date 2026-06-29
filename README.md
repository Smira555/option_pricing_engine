# QuantPricer: Options Pricing & Analytics Engine

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

QuantPricer is an interactive quantitative finance engine for pricing European options and analyzing their risk profiles. It implements analytical closed-form equations (Black-Scholes-Merton), stochastic path simulations (Monte Carlo with variance reduction), and numerical root-finding (Newton-Raphson for Implied Volatility).

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

## Internship Interview Q&A Guide

Prepare for your interviews with these common questions about this codebase:

### Q1: What is the difference between Black-Scholes and Monte Carlo pricing?
*   **Answer**: Black-Scholes is a closed-form analytical model derived from a partial differential equation (PDE) under constant parameters. It is instantaneous and exact but restricted to simple European options. Monte Carlo is a numerical simulation technique that models stock prices along random paths. It is computationally expensive but flexible, allowing pricing of path-dependent options (like Asian or barrier options) or under complex stochastic volatility models.

### Q2: What are the Option Greeks and what do they represent?
*   **Answer**: Greeks measure the option's sensitivity to parameter changes:
    *   **Delta ($\Delta$)**: Sensitivity to underlying price changes ($\frac{\partial V}{\partial S}$). Proxies the probability of expiring in-the-money.
    *   **Gamma ($\Gamma$)**: Sensitivity of Delta to underlying price changes ($\frac{\partial^2 V}{\partial S^2}$). Measures directional risk acceleration.
    *   **Vega ($\nu$)**: Sensitivity to changes in asset volatility ($\frac{\partial V}{\partial \sigma}$).
    *   **Theta ($\Theta$)**: Sensitivity to the passage of time ($\frac{\partial V}{\partial T}$). Represents time decay.
    *   **Rho ($\rho$)**: Sensitivity to changes in the risk-free rate ($\frac{\partial V}{\partial r}$).

### Q3: How do Antithetic Variates work in your Monte Carlo model?
*   **Answer**: Standard Monte Carlo simulation generates independent paths which can suffer from random noise. Antithetic Variates generate pairs of negatively correlated paths ($Z$ and $-Z$). Because the paths are negatively correlated, when one path randomly spikes high, its paired path drops low. Averaging their payoffs cancels out a large portion of the simulation noise (variance), speeding up convergence.

### Q4: How does your Implied Volatility solver work?
*   **Answer**: Implied volatility has no analytical formula. We use the Newton-Raphson method, which starts with a guess ($\sigma_0 = 20\%$) and iteratively updates it. The update step is $\sigma_{n+1} = \sigma_n - \frac{BS(\sigma_n) - C_{market}}{Vega(\sigma_n)}$. Vega acts as the derivative of the pricing function, showing how price changes with volatility. If Newton-Raphson diverges (e.g. due to near-zero Vega), the engine falls back to a Bisection solver which binary-searches the boundary space.

### Q5: Why is this implemented in Python + NumPy instead of C++?
*   **Answer**: In professional quantitative research and trading desks, Python is the industry standard for prototyping, data fetching, and API construction, while compiled languages are reserved for execution backends. By using vectorized NumPy arrays, mathematical calculations are compiled down to optimized C libraries under the hood, making standard simulation tasks nearly as fast as raw C++ while maintaining clean, readable code that is easy to integrate with live data feeds (yfinance) and visual interactive interfaces (Streamlit + Plotly).
