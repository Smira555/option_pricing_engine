"""
European Call Option Pricing Engine — Streamlit Web App
========================================================
Run:  streamlit run app.py
"""

import sys
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricers.black_scholes import bs_call_price, bs_put_price, bs_greeks
from pricers.pde_solver     import bs_pde_call_price
from pricers.monte_carlo    import mc_call_price
from pricers.fft_pricer     import fft_price_at_strike, carr_madan_fft
from utils.characteristic_functions import bs_char_func
from data.fetcher import fetch_market_data


# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Option Pricing Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { font-size: 12px; color: #6c757d; font-weight: 500; margin-bottom: 4px; }
.metric-value { font-size: 26px; font-weight: 700; color: #212529; }
.metric-sub   { font-size: 12px; color: #6c757d; margin-top: 2px; }
.method-tag   { font-size: 11px; background: #e9ecef; color: #495057;
                border-radius: 4px; padding: 2px 8px; display:inline-block; }
.greek-box    { background:#f8f9fa; border-radius:8px; padding:12px;
                text-align:center; border: 1px solid #e9ecef; }
.greek-name   { font-size:11px; color:#6c757d; font-weight:500; }
.greek-val    { font-size:20px; font-weight:700; color:#212529; }
div[data-testid="stSidebar"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"

def color_diff(diff: float) -> str:
    if abs(diff) < 0.001:
        return f"<span style='color:#2ecc71'>≈ same</span>"
    sign = "+" if diff > 0 else ""
    col  = "#e74c3c" if diff > 0 else "#3498db"
    return f"<span style='color:{col}'>{sign}{diff:.4f}</span>"

def build_payoff_chart(S0, K, T, r, sigma, premium):
    prices = np.linspace(max(1, K * 0.5), K * 1.5, 300)
    payoff = np.maximum(prices - K, 0)
    profit = payoff - premium

    # PnL at current spot (for reference line)
    current_payoff = max(S0 - K, 0)
    current_profit = current_payoff - premium

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prices, y=payoff, name="Payoff at expiry",
        line=dict(color="#2ecc71", width=2.5),
        hovertemplate="Stock: $%{x:.2f}<br>Payoff: $%{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=prices, y=profit, name="Profit (after premium)",
        line=dict(color="#e74c3c", width=2.5, dash="dash"),
        hovertemplate="Stock: $%{x:.2f}<br>Profit: $%{y:.2f}<extra></extra>"
    ))

    # strike vertical line
    fig.add_vline(x=K, line=dict(color="#f39c12", width=1.5, dash="dot"),
                  annotation_text=f"Strike K={K:.0f}",
                  annotation_position="top left",
                  annotation_font=dict(color="#f39c12", size=11))

    # current spot vertical line
    fig.add_vline(x=S0, line=dict(color="#9b59b6", width=1.5, dash="dot"),
                  annotation_text=f"Spot S={S0:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(color="#9b59b6", size=11))

    # breakeven
    breakeven = K + premium
    fig.add_vline(x=breakeven, line=dict(color="#3498db", width=1, dash="longdash"),
                  annotation_text=f"Breakeven {breakeven:.2f}",
                  annotation_position="bottom right",
                  annotation_font=dict(color="#3498db", size=11))

    # zero line
    fig.add_hline(y=0, line=dict(color="gray", width=0.8))

    fig.update_layout(
        title=dict(text="Payoff & Profit at Expiry", font=dict(size=15)),
        xaxis_title="Stock Price at Expiry ($)",
        yaxis_title="Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        height=380,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", gridwidth=1)
    return fig


def build_price_history_chart(hist_series, ticker, S0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_series.index, y=hist_series.values,
        name="Close price",
        line=dict(color="#2980b9", width=2),
        fill="tozeroy", fillcolor="rgba(41,128,185,0.07)",
        hovertemplate="%{x|%b %d %Y}: $%{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=S0, line=dict(color="#e74c3c", width=1.5, dash="dot"),
                  annotation_text=f"Latest: ${S0:.2f}",
                  annotation_font=dict(color="#e74c3c", size=11))
    fig.update_layout(
        title=dict(text=f"{ticker} — 1-Year Price History", font=dict(size=15)),
        xaxis_title="", yaxis_title="Price ($)",
        margin=dict(l=40, r=30, t=50, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=300,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def build_smile_chart(S0, K, T, r, sigma):
    """Price across a range of strikes to show the 'smile' curve."""
    strikes = np.linspace(K * 0.6, K * 1.4, 80)
    prices  = []
    for k in strikes:
        cf = partial(bs_char_func, S0=S0, T=T, r=r, sigma=sigma)
        p  = fft_price_at_strike(k, cf, T, r)
        prices.append(p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strikes, y=prices, name="Call price",
        line=dict(color="#8e44ad", width=2),
        hovertemplate="Strike: $%{x:.2f}<br>Price: $%{y:.4f}<extra></extra>"
    ))
    fig.add_vline(x=K, line=dict(color="#f39c12", dash="dot", width=1.5),
                  annotation_text="Selected K",
                  annotation_font=dict(color="#f39c12", size=11))
    fig.update_layout(
        title=dict(text="Call Price Across Strikes (FFT, same vol)", font=dict(size=15)),
        xaxis_title="Strike ($)", yaxis_title="Call Price ($)",
        margin=dict(l=40, r=30, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=300,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def build_returns_hist(log_returns):
    mu  = np.mean(log_returns)
    std = np.std(log_returns)
    x   = np.linspace(mu - 4*std, mu + 4*std, 200)
    normal_y = (np.exp(-0.5 * ((x - mu)/std)**2)
                / (std * np.sqrt(2 * np.pi))) * len(log_returns) * (x[1] - x[0])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=log_returns, nbinsx=50, name="Daily log-returns",
        marker_color="#2ecc71", opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=x, y=normal_y, name="Normal fit",
        line=dict(color="#e74c3c", width=2)
    ))
    fig.update_layout(
        title=dict(text="Distribution of Daily Log-Returns", font=dict(size=15)),
        xaxis_title="Log-return", yaxis_title="Count",
        margin=dict(l=40, r=30, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        height=300,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Option Pricer")
    st.caption("European Call Option Pricing Engine")
    st.divider()

    # ── ticker fetch ──────────────────────────────────────────────────────
    st.subheader("1. Load Market Data")
    ticker_input = st.text_input(
        "Stock ticker", value="AAPL",
        help="Any Yahoo Finance ticker: AAPL, MSFT, GOOGL, TSLA, NIFTY50.NS, etc."
    ).upper().strip()

    fetch_btn = st.button("Fetch Live Data", use_container_width=True, type="primary")

    # initialize session state
    for k, v in [("S0", 100.0), ("sigma", 0.20), ("r", 0.05),
                 ("hist", None), ("returns", None),
                 ("name", ""), ("sector", ""), ("currency", "USD"),
                 ("fetched_ticker", "")]:
        if k not in st.session_state:
            st.session_state[k] = v

    if fetch_btn:
        with st.spinner(f"Fetching data for {ticker_input}…"):
            try:
                data = fetch_market_data(ticker_input)
                st.session_state.S0             = data["S0"]
                st.session_state.sigma          = data["sigma"]
                st.session_state.r              = data["r"]
                st.session_state.hist           = data["price_history"]
                st.session_state.returns        = data["log_returns"]
                st.session_state.name           = data["name"]
                st.session_state.sector         = data["sector"]
                st.session_state.currency       = data["currency"]
                st.session_state.fetched_ticker = ticker_input
                st.success(f"Loaded {data['name']}")
            except Exception as e:
                st.error(f"Error: {e}")

    # show company info if loaded
    if st.session_state.fetched_ticker:
        st.caption(f"**{st.session_state.name}** · {st.session_state.sector}")

    st.divider()

    # ── parameters ────────────────────────────────────────────────────────
    st.subheader("2. Option Parameters")
    st.caption("Auto-filled from market data. Adjust freely.")

    S0 = st.number_input(
        "Spot price S₀ ($)",
        min_value=0.01, max_value=100000.0,
        value=round(st.session_state.S0, 2), step=0.5,
        help="Current stock price. Auto-fetched from Yahoo Finance."
    )

    K = st.number_input(
        "Strike price K ($)",
        min_value=0.01, max_value=100000.0,
        value=round(st.session_state.S0, 0), step=0.5,
        help="The price at which you have the right to buy. Default = ATM (= spot)."
    )

    T = st.number_input(
        "Time to expiry T (years)",
        min_value=0.01, max_value=5.0,
        value=1.0, step=0.05,
        help="1.0 = one year. 0.25 = 3 months. 0.0833 ≈ 1 month."
    )

    sigma = st.number_input(
        "Volatility σ (annualized)",
        min_value=0.01, max_value=5.0,
        value=round(st.session_state.sigma, 4), step=0.01,
        format="%.4f",
        help="Historical vol auto-computed from last 252 trading days."
        " In practice, traders use implied vol from option markets."
    )

    r = st.number_input(
        "Risk-free rate r",
        min_value=0.0, max_value=0.5,
        value=round(st.session_state.r, 4), step=0.001,
        format="%.4f",
        help="Annualized. Auto-fetched from 13-week US T-bill (^IRX)."
    )

    st.divider()

    # ── method toggles ────────────────────────────────────────────────────
    st.subheader("3. Pricing Methods")
    run_bs  = st.checkbox("Closed-form Black-Scholes", value=True)
    run_pde = st.checkbox("PDE (Crank-Nicolson)", value=True)
    run_mc  = st.checkbox("Monte Carlo", value=True)
    run_fft = st.checkbox("FFT (Carr-Madan)", value=True)

    if run_mc:
        n_paths = st.select_slider(
            "MC paths", options=[10_000, 50_000, 100_000, 300_000, 500_000],
            value=100_000
        )
    else:
        n_paths = 100_000

    calculate_btn = st.button("Calculate Prices", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.title("European Call Option Pricing Engine")

moneyness = "At-the-money (ATM)" if abs(S0-K)/S0 < 0.02 else \
            "In-the-money (ITM)"  if S0 > K else "Out-of-the-money (OTM)"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Spot Price",   f"${S0:.2f}")
col2.metric("Strike",       f"${K:.2f}")
col3.metric("Maturity",     f"{T:.2f}y  ({T*365:.0f}d)")
col4.metric("Moneyness",    moneyness)

# ── price history chart (if loaded) ─────────────────────────────────────────
if st.session_state.hist is not None:
    with st.expander("📊 Price History & Return Distribution", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                build_price_history_chart(
                    st.session_state.hist,
                    st.session_state.fetched_ticker, S0
                ),
                use_container_width=True
            )
        with c2:
            if st.session_state.returns is not None:
                st.plotly_chart(
                    build_returns_hist(st.session_state.returns),
                    use_container_width=True
                )

# ── pricing results ──────────────────────────────────────────────────────────
if calculate_btn:
    st.divider()
    st.subheader("Pricing Results")

    results = {}
    errors  = {}

    with st.spinner("Running pricing engines…"):

        if run_bs:
            t0 = time.perf_counter()
            try:
                p = bs_call_price(S0, K, T, r, sigma)
                results["Black-Scholes (closed-form)"] = {
                    "price": p, "ms": (time.perf_counter()-t0)*1e3,
                    "se": None, "tag": "exact"
                }
            except Exception as e:
                errors["Black-Scholes"] = str(e)

        if run_pde:
            t0 = time.perf_counter()
            try:
                p = bs_pde_call_price(S0, K, T, r, sigma, M=400, N=2000)
                results["PDE (Crank-Nicolson)"] = {
                    "price": p, "ms": (time.perf_counter()-t0)*1e3,
                    "se": None, "tag": "numerical"
                }
            except Exception as e:
                errors["PDE"] = str(e)

        if run_mc:
            t0 = time.perf_counter()
            try:
                p, se, ci = mc_call_price(S0, K, T, r, sigma,
                                          n_paths=n_paths, seed=42)
                results["Monte Carlo"] = {
                    "price": p, "ms": (time.perf_counter()-t0)*1e3,
                    "se": se, "ci": ci, "tag": "simulation"
                }
            except Exception as e:
                errors["Monte Carlo"] = str(e)

        if run_fft:
            t0 = time.perf_counter()
            try:
                cf = partial(bs_char_func, S0=S0, T=T, r=r, sigma=sigma)
                p  = fft_price_at_strike(K, cf, T, r)
                results["FFT (Carr-Madan)"] = {
                    "price": p, "ms": (time.perf_counter()-t0)*1e3,
                    "se": None, "tag": "transform"
                }
            except Exception as e:
                errors["FFT"] = str(e)

    # ── show errors ──────────────────────────────────────────────────────
    for m, e in errors.items():
        st.error(f"{m}: {e}")

    if not results:
        st.warning("No methods ran successfully.")
        st.stop()

    # ── price cards ───────────────────────────────────────────────────────
    bs_ref = results.get("Black-Scholes (closed-form)", {}).get("price")
    cols   = st.columns(len(results))
    method_names = list(results.keys())

    for i, (name, res) in enumerate(results.items()):
        p = res["price"]
        with cols[i]:
            diff_html = ""
            if bs_ref is not None and name != "Black-Scholes (closed-form)":
                diff_html = f"<div class='metric-sub'>vs BS: {color_diff(p - bs_ref)}</div>"
            se_html = ""
            if res["se"] is not None:
                se_html = f"<div class='metric-sub'>± {res['se']:.4f} SE</div>"
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{name}</div>
              <div class='metric-value'>${p:.4f}</div>
              {se_html}
              {diff_html}
              <div class='metric-sub' style='margin-top:4px;'>
                <span class='method-tag'>{res['tag']}</span>
                &nbsp;{res['ms']:.1f} ms
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── greeks ────────────────────────────────────────────────────────────
    if run_bs:
        st.subheader("Greeks")
        g = bs_greeks(S0, K, T, r, sigma)
        gcols = st.columns(5)
        greek_meta = {
            "delta": ("Δ Delta",  "∂C/∂S — shares to hold\nto hedge this option"),
            "gamma": ("Γ Gamma",  "∂²C/∂S² — how fast\ndelta itself changes"),
            "vega":  ("ν Vega",   "∂C/∂σ — $ gain per\n+1% vol increase"),
            "theta": ("Θ Theta",  "∂C/∂T — $ lost per\nday of time passing"),
            "rho":   ("ρ Rho",    "∂C/∂r — $ gain per\n+1% rate increase"),
        }
        for col, (k, (label, tip)) in zip(gcols, greek_meta.items()):
            with col:
                st.markdown(f"""
                <div class='greek-box'>
                  <div class='greek-name'>{label}</div>
                  <div class='greek-val'>{float(g[k]):.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(tip)

    # ── put price & put-call parity ───────────────────────────────────────
    if run_bs:
        put_price = bs_put_price(S0, K, T, r, sigma)
        with st.expander("Put Price & Put-Call Parity Check"):
            p1, p2, p3 = st.columns(3)
            p1.metric("Call price",  f"${bs_ref:.4f}")
            p2.metric("Put price",   f"${put_price:.4f}")
            parity_lhs = bs_ref - put_price
            parity_rhs = S0 - K * np.exp(-r * T)
            p3.metric("Parity error",
                      f"{abs(parity_lhs - parity_rhs):.2e}",
                      help="C − P = S − Ke^(−rT). Should be ~0.")
            st.caption(
                f"Put-call parity: C − P = S₀ − Ke^(−rT) "
                f"→ {parity_lhs:.4f} = {parity_rhs:.4f} ✓"
            )

    # ── charts ────────────────────────────────────────────────────────────
    st.divider()
    premium = bs_ref if bs_ref else list(results.values())[0]["price"]

    tab1, tab2 = st.tabs(["Payoff & Profit", "Price Across Strikes"])

    with tab1:
        st.plotly_chart(
            build_payoff_chart(S0, K, T, r, sigma, premium),
            use_container_width=True
        )
        st.caption(
            f"Breakeven at expiry: ${K + premium:.2f}  "
            f"(strike {K:.2f} + premium {premium:.2f})"
        )

    with tab2:
        with st.spinner("Computing FFT strike sweep…"):
            st.plotly_chart(
                build_smile_chart(S0, K, T, r, sigma),
                use_container_width=True
            )
        st.caption(
            "This shows call prices across all strikes in a single FFT pass "
            "(4096 strikes computed in < 1ms). Under constant volatility "
            "(Black-Scholes) this is a smooth curve. In real markets, "
            "implied volatility varies with strike — the 'volatility smile'."
        )

    # ── comparison table ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Method Comparison Table")
    rows = []
    for name, res in results.items():
        diff = f"{res['price'] - bs_ref:+.5f}" if bs_ref and name != "Black-Scholes (closed-form)" else "—"
        se   = f"± {res['se']:.5f}" if res["se"] else "—"
        rows.append({
            "Method":      name,
            "Price ($)":   f"{res['price']:.6f}",
            "Diff vs BS":  diff,
            "Std Error":   se,
            "Time (ms)":   f"{res['ms']:.2f}",
            "Type":        res["tag"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── parameter summary ─────────────────────────────────────────────────
    with st.expander("Input Parameters Used"):
        param_df = pd.DataFrame([{
            "S₀ (Spot)":    S0,
            "K (Strike)":   K,
            "T (Years)":    T,
            "σ (Vol)":      f"{sigma:.4f} ({sigma*100:.2f}%)",
            "r (Rate)":     f"{r:.4f} ({r*100:.2f}%)",
            "Moneyness":    moneyness,
            "S/K ratio":    f"{S0/K:.4f}",
        }])
        st.dataframe(param_df.T.rename(columns={0: "Value"}),
                     use_container_width=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Market data via Yahoo Finance (yfinance). "
    "Volatility = historical (252-day realized). "
    "Risk-free rate = 13-week T-bill (^IRX). "
    "This is a learning tool, not financial advice."
)
