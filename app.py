import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.engine.black_scholes import black_scholes_price, black_scholes_greeks
from backend.engine.monte_carlo import monte_carlo_price
from backend.engine.implied_vol import calculate_implied_volatility
from backend.engine.data_fetcher import get_stock_data

# Page config
st.set_page_config(
    page_title="QuantPricer | Option Pricing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .greek-box {
        background-color: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .greek-symbol {
        font-size: 1.8rem;
        font-weight: bold;
        color: #7d5fff;
    }
    .header-logo {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7d5fff 0%, #00d2d3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States for inputs
if "spot_price" not in st.session_state:
    st.session_state.spot_price = 100.0
if "volatility" not in st.session_state:
    st.session_state.volatility = 20.0
if "company_name" not in st.session_state:
    st.session_state.company_name = ""
if "data_source" not in st.session_state:
    st.session_state.data_source = "Manual"

# Title Area
st.markdown('<span class="header-logo">QuantPricer</span>', unsafe_allow_html=True)
st.markdown("### European Option Pricing & Risk Analysis Suite")
st.write("---")

# Sidebar Configuration
st.sidebar.header("Asset & Option Configuration")

# Market mode toggle
market_mode = st.sidebar.checkbox("Load Live Market Data", value=False)

if market_mode:
    ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL", help="Enter AAPL, TSLA, MSFT, etc.")
    if st.sidebar.button("Fetch Live Data"):
        with st.spinner("Fetching stock metrics..."):
            stock_info = get_stock_data(ticker)
            if "current_price" in stock_info:
                st.session_state.spot_price = float(stock_info["current_price"])
                st.session_state.volatility = float(stock_info["historical_volatility"] * 100)
                st.session_state.company_name = stock_info["company_name"]
                st.session_state.data_source = stock_info["source"]
                st.sidebar.success(f"Loaded: {stock_info['company_name']}")
            else:
                st.sidebar.error("Could not fetch ticker data.")

# Set widgets linked to session state or standard controls
if market_mode:
    st.sidebar.info(f"Asset: **{st.session_state.company_name}**\n\nPrice: **${st.session_state.spot_price:.2f}**\n\nVolatility: **{st.session_state.volatility:.2f}%**\n\nSource: *{st.session_state.data_source}*")
    S = st.session_state.spot_price
    sigma_percent = st.session_state.volatility
else:
    S = st.sidebar.number_input("Underlying Stock Price (S)", value=100.0, step=1.0, min_value=0.01)
    sigma_percent = st.sidebar.number_input("Annualized Volatility (σ %)", value=20.0, step=1.0, min_value=0.1)

K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0, min_value=0.01)
days_to_maturity = st.sidebar.number_input("Days to Expiry (T)", value=365, step=1, min_value=1)
T = days_to_maturity / 365.0

r_percent = st.sidebar.number_input("Risk-Free Interest Rate (r %)", value=5.00, step=0.1)
r = r_percent / 100.0
sigma = sigma_percent / 100.0

option_type = st.sidebar.radio("Option Type", ["Call", "Put"]).lower()

# Advanced Section
st.sidebar.subheader("Simulation Parameters")
num_simulations = st.sidebar.selectbox("Monte Carlo Paths", [1000, 10000, 50000, 100000], index=2)
use_antithetic = st.sidebar.checkbox("Use Antithetic Variates", value=True, help="Variance reduction technique")

market_price_str = st.sidebar.text_input("Observed Option Market Price ($)", value="", help="Leave blank if unknown. Enter price to solve for Implied Volatility.")
market_price = float(market_price_str) if market_price_str.strip() != "" else None

# Calculations
with st.spinner("Executing analytical models and path simulations..."):
    # 1. Closed-form BS pricing
    bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
    
    # 2. Greeks
    greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
    
    # 3. Monte Carlo
    mc_results = monte_carlo_price(S, K, T, r, sigma, option_type, num_simulations, use_antithetic)
    
    # 4. Solved Implied Volatility
    implied_vol = None
    if market_price is not None:
        implied_vol = calculate_implied_volatility(S, K, T, r, market_price, option_type)

# Main Grid Layout
tabs = st.tabs(["Valuation & Greeks", "Monte Carlo Simulations", "Sensitivity Curves"])

# Tab 1: Valuation
with tabs[0]:
    # 2 columns for major pricing cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Black-Scholes Price</h4>
            <h1 style="color: #7d5fff; font-size: 3.5rem;">${bs_price:.4f}</h1>
            <p style="color: #8b949e; font-size: 0.85rem;">Analytical formula exact solution</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Monte Carlo Price</h4>
            <h1 style="color: #00d2d3; font-size: 3.5rem;">${mc_results['price']:.4f}</h1>
            <p style="color: #8b949e; font-size: 0.85rem;">Confidence bounds: [{mc_results['ci_lower']:.4f} - {mc_results['ci_upper']:.4f}]</p>
        </div>
        """, unsafe_allow_html=True)

    # Implied Volatility (If calculated)
    if implied_vol is not None:
        st.write("")
        st.markdown(f"""
        <div style="background-color: rgba(125, 95, 255, 0.1); border: 1px solid rgba(125, 95, 255, 0.3); border-radius: 12px; padding: 1.2rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="color: #a29bfe; margin:0;">Solved Implied Volatility</h4>
                <p style="color: #8b949e; font-size: 0.8rem; margin:0;">Calculated via Newton-Raphson from market price of ${market_price:.2f}</p>
            </div>
            <div style="text-align: right;">
                <h2 style="color: #a29bfe; margin:0; font-size: 2.2rem;">{implied_vol*100:.2f}%</h2>
                <p style="color: #8b949e; font-size: 0.8rem; margin:0;">vs. {sigma*100:.1f}% Model Input Vol</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Analytical Option Greeks Suite")
    st.write("Click hover question marks in details to read their exact definitions.")
    
    # 5 Columns for greeks
    g_cols = st.columns(5)
    
    g_data = [
        ("Delta (Δ)", greeks["delta"], "Sensitivity to underlying stock price. Call delta ranges from 0 to 1, acting as a probability proxy of expiring ITM."),
        ("Gamma (Γ)", greeks["gamma"], "Rate of change of Delta. Highest for At-The-Money options, indicating speed of directional risk changes."),
        ("Vega (ν)", greeks["vega"], "Sensitivity of option price to a 1% shift in underlying volatility. High volatility increases premium value."),
        ("Theta (Θ)", greeks["theta_per_day"], "Option price decay rate per day. Accelerates as the option approaches maturity."),
        ("Rho (ρ)", greeks["rho"], "Sensitivity of option price to a 1% change in risk-free interest rates.")
    ]
    
    for idx, (name, val, tooltip) in enumerate(g_data):
        with g_cols[idx]:
            # Apply color to Delta
            val_color = "#f5f6fa"
            if name.startswith("Delta"):
                val_color = "#05c46b" if val > 0 else "#ff3f34"
                
            st.markdown(f"""
            <div class="greek-box">
                <p style="color: #8b949e; font-size: 0.8rem; text-transform: uppercase; font-weight: 600; margin-bottom: 0.2rem;">{name}</p>
                <h2 style="color: {val_color}; font-size: 1.8rem; margin-bottom: 0.2rem;">{val:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
            st.info(tooltip)

# Tab 2: Monte Carlo Paths & Convergence
with tabs[1]:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Stochastic Path Trajectories")
        st.write("Visualizing 10 independent simulated stock price paths under Geometric Brownian Motion (GBM).")
        
        # Plotly path generation
        time_grid = np.array(mc_results["time_grid"]) * 365
        paths = mc_results["paths"]
        
        fig_paths = go.Figure()
        for idx, path in enumerate(paths):
            fig_paths.add_trace(go.Scatter(
                x=time_grid, y=path,
                mode='lines',
                name=f'Path {idx+1}',
                line=dict(width=1.5)
            ))
        fig_paths.update_layout(
            template="plotly_dark",
            xaxis_title="Time to Expiry (Days)",
            yaxis_title="Stock Price ($)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_chart2:
        st.subheader("Monte Carlo Convergence Rate")
        st.write("Plotting running simulation average price against exact Black-Scholes baseline.")
        
        convergence_data = mc_results["convergence"]
        cps = [item["paths"] for item in convergence_data]
        mc_prices = [item["price"] for item in convergence_data]
        
        fig_conv = go.Figure()
        # MC price
        fig_conv.add_trace(go.Scatter(
            x=cps, y=mc_prices,
            mode='lines+markers',
            name='Monte Carlo Price',
            line=dict(color='#00d2d3', width=2.5),
            marker=dict(size=6)
        ))
        # BS baseline
        fig_conv.add_trace(go.Scatter(
            x=cps, y=[bs_price]*len(cps),
            mode='lines',
            name='Analytical BS Price',
            line=dict(color='#7d5fff', dash='dash', width=2)
        ))
        fig_conv.update_layout(
            template="plotly_dark",
            xaxis_title="Number of Paths Simulated",
            yaxis_title="Option Price ($)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_conv, use_container_width=True)

# Tab 3: Sensitivities
with tabs[2]:
    st.subheader("Price Sensitivity to Stock price shifts")
    st.write("Plots the European option valuation and its intrinsic payoff across stock price changes of -20% to +20%.")
    
    # Generate sensitivity line points
    spot_range = np.linspace(S * 0.8, S * 1.2, 30)
    option_prices = []
    payoffs = []
    
    # Simple mathematical BS price calculation inline for mapping
    # To keep code decoupling clean, let's use the core BS pricing function
    for s_step in spot_range:
        price_step = black_scholes_price(s_step, K, T, r, sigma, option_type)
        option_prices.append(price_step)
        
        payoff_step = max(s_step - K, 0.0) if option_type == "call" else max(K - s_step, 0.0)
        payoffs.append(payoff_step)
        
    fig_sens = go.Figure()
    # Option value curve
    curve_color = '#05c46b' if option_type == 'call' else '#ff3f34'
    fig_sens.add_trace(go.Scatter(
        x=spot_range, y=option_prices,
        mode='lines',
        name='Theoretical Option Value',
        line=dict(color=curve_color, width=3.5),
        fill='tozeroy'
    ))
    # Payoff curve
    fig_sens.add_trace(go.Scatter(
        x=spot_range, y=payoffs,
        mode='lines',
        name='Intrinsic Payoff (At Expiration)',
        line=dict(color='rgba(255,255,255,0.4)', dash='dot', width=1.5)
    ))
    
    fig_sens.update_layout(
        template="plotly_dark",
        xaxis_title="Underlying Stock Price ($)",
        yaxis_title="Option Valuation ($)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_sens, use_container_width=True)
