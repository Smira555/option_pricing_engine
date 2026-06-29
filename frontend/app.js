// Global State
let chartInstance = null;
let lastCalculationData = null;
let activeTab = "paths"; // default tab
let apiBaseUrl = ""; // empty string resolves to current host in relative requests

// DOM Elements
const manualModeBtn = document.getElementById("manualModeBtn");
const marketModeBtn = document.getElementById("marketModeBtn");
const tickerSection = document.getElementById("tickerSection");
const tickerInput = document.getElementById("tickerInput");
const fetchTickerBtn = document.getElementById("fetchTickerBtn");
const tickerInfoMessage = document.getElementById("tickerInfoMessage");

const pricingForm = document.getElementById("pricingForm");
const spotPriceInput = document.getElementById("spotPrice");
const strikePriceInput = document.getElementById("strikePrice");
const maturityDaysInput = document.getElementById("maturityDays");
const maturityYearsText = document.getElementById("maturityYearsText");
const riskFreeRateInput = document.getElementById("riskFreeRate");
const volatilityInput = document.getElementById("volatility");
const volatilitySourceText = document.getElementById("volatilitySourceText");
const callOptionRadio = document.getElementById("callOption");
const putOptionRadio = document.getElementById("putOption");

// Advanced config
const numSimulationsSelect = document.getElementById("numSimulations");
const useAntitheticCheckbox = document.getElementById("useAntithetic");
const marketPriceInput = document.getElementById("marketPrice");

// Outputs
const calculateBtn = document.getElementById("calculateBtn");
const btnSpinner = document.getElementById("btnSpinner");
const statusText = document.getElementById("statusText");

const bsPriceDisplay = document.getElementById("bsPriceDisplay");
const mcPriceDisplay = document.getElementById("mcPriceDisplay");
const mcCiDisplay = document.getElementById("mcCiDisplay");
const ivCard = document.getElementById("ivCard");
const ivDisplay = document.getElementById("ivDisplay");
const ivComparison = document.getElementById("ivComparison");

const deltaDisplay = document.getElementById("deltaDisplay");
const gammaDisplay = document.getElementById("gammaDisplay");
const vegaDisplay = document.getElementById("vegaDisplay");
const thetaDisplay = document.getElementById("thetaDisplay");
const rhoDisplay = document.getElementById("rhoDisplay");

// Tabs
const tabPathsBtn = document.getElementById("tabPathsBtn");
const tabConvergenceBtn = document.getElementById("tabConvergenceBtn");
const tabSensitivityBtn = document.getElementById("tabSensitivityBtn");
const chartCaption = document.getElementById("chartCaption");

// Initialization
document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    updateMaturityYears();
    // Run initial pricing with default form values
    calculateOptionPricing();
});

// Event Listeners Setup
function setupEventListeners() {
    // Mode Selection (Manual vs. Live Market Data)
    manualModeBtn.addEventListener("click", () => {
        manualModeBtn.classList.add("active");
        marketModeBtn.classList.remove("active");
        tickerSection.classList.add("hidden");
        volatilitySourceText.innerText = "Annual standard deviation of returns";
        volatilityInput.readOnly = false;
        spotPriceInput.readOnly = false;
    });

    marketModeBtn.addEventListener("click", () => {
        marketModeBtn.classList.add("active");
        manualModeBtn.classList.remove("active");
        tickerSection.classList.remove("hidden");
    });

    // Fetch Stock Volatility and Price from API
    fetchTickerBtn.addEventListener("click", fetchStockMarketData);
    tickerInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            fetchStockMarketData();
        }
    });

    // Maturity day auto-converter
    maturityDaysInput.addEventListener("input", updateMaturityYears);

    // Form submit for calculation
    pricingForm.addEventListener("submit", (e) => {
        e.preventDefault();
        calculateOptionPricing();
    });

    // Chart Tabs
    tabPathsBtn.addEventListener("click", () => switchChartTab("paths"));
    tabConvergenceBtn.addEventListener("click", () => switchChartTab("convergence"));
    tabSensitivityBtn.addEventListener("click", () => switchChartTab("sensitivity"));
}

// Update display text for time in years
function updateMaturityYears() {
    const days = parseFloat(maturityDaysInput.value) || 0;
    const years = (days / 365).toFixed(2);
    maturityYearsText.innerText = `(= ${years} Year${years === "1.00" ? "" : "s"})`;
}

// Switch visual chart tab
function switchChartTab(tabName) {
    activeTab = tabName;
    
    // Update active class on tab buttons
    [tabPathsBtn, tabConvergenceBtn, tabSensitivityBtn].forEach(btn => btn.classList.remove("active"));
    
    if (tabName === "paths") {
        tabPathsBtn.classList.add("active");
        chartCaption.innerText = "Visualizing 10 independent stock price paths simulated over time via Geometric Brownian Motion (GBM).";
    } else if (tabName === "convergence") {
        tabConvergenceBtn.classList.add("active");
        chartCaption.innerText = "Shows how the Monte Carlo price estimate converges to the exact Black-Scholes price (dotted line) as the number of simulated paths increases.";
    } else if (tabName === "sensitivity") {
        tabSensitivityBtn.classList.add("active");
        chartCaption.innerText = "Sensitivity curve plotting option price against varying spot prices of the underlying asset (ranging from -20% to +20% of current price).";
    }

    if (lastCalculationData) {
        renderChart();
    }
}

// Fetch yfinance data from backend API
async function fetchStockMarketData() {
    const ticker = tickerInput.value.trim();
    if (!ticker) return;

    fetchTickerBtn.disabled = true;
    tickerInfoMessage.className = "ticker-info-msg";
    tickerInfoMessage.innerText = "Connecting to market server...";

    try {
        const response = await fetch(`${apiBaseUrl}/api/stock?ticker=${ticker}`);
        if (!response.ok) {
            throw new Error(`Ticker ${ticker} not found or network error.`);
        }
        
        const data = await response.json();
        
        // Populate inputs with live data
        spotPriceInput.value = data.current_price.toFixed(2);
        volatilityInput.value = (data.historical_volatility * 100).toFixed(2);
        
        // Update read-only properties since market mode is active
        volatilityInput.readOnly = true;
        spotPriceInput.readOnly = true;

        // Display success banner details
        volatilitySourceText.innerText = `Computed volatility: ${volatilityInput.value}% (Source: ${data.source})`;
        tickerInfoMessage.className = "ticker-info-msg";
        tickerInfoMessage.innerText = `Success: Loaded ${data.company_name} ($${data.current_price.toFixed(2)})`;
        
        statusText.innerText = "Updated with live market data";
        
        // Auto-run calculator with new values
        calculateOptionPricing();

    } catch (error) {
        tickerInfoMessage.className = "ticker-info-msg error";
        tickerInfoMessage.innerText = error.message;
    } finally {
        fetchTickerBtn.disabled = false;
    }
}

// Call API to perform option pricing and greeks calculation
async function calculateOptionPricing() {
    // Collect parameters
    const S = parseFloat(spotPriceInput.value);
    const K = parseFloat(strikePriceInput.value);
    const days = parseFloat(maturityDaysInput.value);
    const T = days / 365.0; // Time in years
    const r = parseFloat(riskFreeRateInput.value) / 100.0; // convert % to decimal
    const sigma = parseFloat(volatilityInput.value) / 100.0; // convert % to decimal
    const option_type = callOptionRadio.checked ? "call" : "put";

    const num_simulations = parseInt(numSimulationsSelect.value);
    const use_antithetic = useAntitheticCheckbox.checked;
    
    const marketPriceVal = marketPriceInput.value.trim();
    const market_price = marketPriceVal !== "" ? parseFloat(marketPriceVal) : null;

    // UI Loading state
    calculateBtn.disabled = true;
    btnSpinner.classList.remove("hidden");
    statusText.innerText = "Pricing Option...";

    try {
        const payload = {
            S, K, T, r, sigma, option_type,
            num_simulations, use_antithetic, market_price
        };

        const response = await fetch(`${apiBaseUrl}/api/price`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server error running pricing calculation.");
        }

        const data = await response.json();
        lastCalculationData = data;

        // Display results
        updateResultUI(data, option_type, sigma);
        statusText.innerText = "Calculation Complete";

    } catch (error) {
        console.error(error);
        statusText.innerText = `Error: ${error.message}`;
    } finally {
        calculateBtn.disabled = false;
        btnSpinner.classList.add("hidden");
    }
}

// Update DOM elements with calculated pricing results
function updateResultUI(data, option_type, inputSigma) {
    // 1. Set Prices
    bsPriceDisplay.innerText = data.analytical.price.toFixed(4);
    mcPriceDisplay.innerText = data.monte_carlo.price.toFixed(4);
    
    // 2. Set Confidence Interval
    const ci_lower = data.monte_carlo.ci_lower.toFixed(4);
    const ci_upper = data.monte_carlo.ci_upper.toFixed(4);
    const std_err = data.monte_carlo.std_err.toFixed(5);
    mcCiDisplay.innerText = `95% CI: [${ci_lower} - ${ci_upper}] (Std. Error: ${std_err})`;

    // 3. Set Greeks
    const greeks = data.analytical.greeks;
    deltaDisplay.innerText = greeks.delta.toFixed(4);
    gammaDisplay.innerText = greeks.gamma.toFixed(4);
    vegaDisplay.innerText = greeks.vega.toFixed(4);
    thetaDisplay.innerText = greeks.theta_per_day.toFixed(4); // Display daily theta decay
    rhoDisplay.innerText = greeks.rho.toFixed(4);

    // Color code delta according to positive/negative values
    if (greeks.delta > 0) {
        deltaDisplay.style.color = "var(--success)";
    } else {
        deltaDisplay.style.color = "var(--danger)";
    }

    // 4. Implied Volatility Card
    if (data.implied_volatility !== null) {
        ivCard.classList.remove("hidden");
        const solvedIvPercent = (data.implied_volatility * 100).toFixed(2);
        ivDisplay.innerText = `${solvedIvPercent}%`;
        
        const inputVolPercent = (inputSigma * 100).toFixed(1);
        ivComparison.innerText = `vs. ${inputVolPercent}% Asset Vol`;
    } else {
        ivCard.classList.add("hidden");
    }

    // 5. Render active chart
    renderChart();
}

// Render selected chart using Chart.js
function renderChart() {
    if (!lastCalculationData) return;

    const ctx = document.getElementById("dashboardChart").getContext("2d");
    
    // Destroy existing chart to rebuild layout clean
    if (chartInstance) {
        chartInstance.destroy();
    }

    if (activeTab === "paths") {
        renderPathsChart(ctx);
    } else if (activeTab === "convergence") {
        renderConvergenceChart(ctx);
    } else if (activeTab === "sensitivity") {
        renderSensitivityChart(ctx);
    }
}

// Render Monte Carlo Paths (GBM random walks)
function renderPathsChart(ctx) {
    const paths = lastCalculationData.monte_carlo.paths;
    const timeGrid = lastCalculationData.monte_carlo.time_grid;
    
    // Convert time grid to friendly step labels (in years or days)
    const labels = timeGrid.map(t => (t * 365).toFixed(0) + "d");

    const datasets = paths.map((path, idx) => {
        // Generate distinct neon colors with fade
        const hue = (idx * 36) % 360;
        return {
            label: `Path ${idx + 1}`,
            data: path,
            borderColor: `hsla(${hue}, 85%, 65%, 0.8)`,
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0.1
        };
    });

    chartInstance = new Chart(ctx, {
        type: "line",
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false } // Hide legend to avoid cluttering 10 lines
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    title: { display: true, text: "Time to Expiry (Days)", color: "var(--text-muted)" },
                    ticks: { color: "var(--text-muted)" }
                },
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    title: { display: true, text: "Asset Price ($)", color: "var(--text-muted)" },
                    ticks: { color: "var(--text-muted)" }
                }
            }
        }
    });
}

// Render Monte Carlo Convergence over paths count
function renderConvergenceChart(ctx) {
    const convergence = lastCalculationData.monte_carlo.convergence;
    const bsPrice = lastCalculationData.analytical.price;

    const labels = convergence.map(item => item.paths.toLocaleString());
    const mcPrices = convergence.map(item => item.price);
    
    // Create an array matching labels filled with BS price for the horizontal comparison line
    const bsPricesArray = Array(labels.length).fill(bsPrice);

    chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Monte Carlo Running Estimate",
                    data: mcPrices,
                    borderColor: "#00d2d3",
                    backgroundColor: "rgba(0, 210, 211, 0.1)",
                    borderWidth: 2.5,
                    pointBackgroundColor: "#00d2d3",
                    pointRadius: 4,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: "Analytical Black-Scholes Price",
                    data: bsPricesArray,
                    borderColor: "var(--primary)",
                    borderDash: [5, 5],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "var(--text-main)", font: { family: "Outfit" } }
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    title: { display: true, text: "Number of Simulations", color: "var(--text-muted)" },
                    ticks: { color: "var(--text-muted)" }
                },
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    title: { display: true, text: "Option Price ($)", color: "var(--text-muted)" },
                    ticks: { color: "var(--text-muted)" }
                }
            }
        }
    });
}

// Render Option Price Sensitivity (Spot Price vs Option Price)
function renderSensitivityChart(ctx) {
    // Generate data points by calling BS pricing client-side
    // We vary the spot price from 80% to 120% of current input price
    const S_current = parseFloat(spotPriceInput.value);
    const K = parseFloat(strikePriceInput.value);
    const T = parseFloat(maturityDaysInput.value) / 365.0;
    const r = parseFloat(riskFreeRateInput.value) / 100.0;
    const sigma = parseFloat(volatilityInput.value) / 100.0;
    const option_type = callOptionRadio.checked ? "call" : "put";

    const labels = [];
    const bsPrices = [];
    const intrinsicValues = [];

    // 20 steps from 80% to 120% of stock price
    for (let i = 0; i <= 20; i++) {
        const factor = 0.8 + (i * 0.4) / 20;
        const S_step = S_current * factor;
        labels.push(S_step.toFixed(1));
        
        // Calculate BS price
        const price = jsBlackScholes(S_step, K, T, r, sigma, option_type);
        bsPrices.push(price);

        // Intrinsic value payoff at maturity (max(S-K,0) or max(K-S,0))
        const intrinsic = option_type === "call" ? Math.max(S_step - K, 0) : Math.max(K - S_step, 0);
        intrinsicValues.push(intrinsic);
    }

    chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Option Theoretical Value (Black-Scholes)",
                    data: bsPrices,
                    borderColor: option_type === "call" ? "var(--success)" : "var(--danger)",
                    backgroundColor: option_type === "call" ? "rgba(5, 196, 107, 0.05)" : "rgba(255, 63, 52, 0.05)",
                    borderWidth: 3,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: "Intrinsic Payoff (At Expiration)",
                    data: intrinsicValues,
                    borderColor: "rgba(255, 255, 255, 0.25)",
                    borderDash: [3, 3],
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "var(--text-main)", font: { family: "Outfit" } }
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    title: { display: true, text: "Underlying Stock Price ($)", color: "var(--text-muted)" },
                    ticks: { color: "var(--text-muted)" }
                },
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    title: { display: true, text: "Option Price ($)", color: "var(--text-muted)" },
                    ticks: { color: "var(--text-muted)" }
                }
            }
        }
    });
}

// Client-side analytical Black-Scholes formula for instant sensitivity graphs
function jsBlackScholes(S, K, T, r, sigma, type) {
    if (T <= 0) {
        return type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
    }
    
    if (sigma <= 0) {
        const discountedK = K * Math.exp(-r * T);
        return type === "call" ? Math.max(S - discountedK, 0) : Math.max(discountedK - S, 0);
    }

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);

    if (type === "call") {
        return S * stdNormalCDF(d1) - K * Math.exp(-r * T) * stdNormalCDF(d2);
    } else {
        return K * Math.exp(-r * T) * stdNormalCDF(-d2) - S * stdNormalCDF(-d1);
    }
}

// Standard Normal CDF Approximation (extremely precise)
function stdNormalCDF(x) {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    const absX = Math.abs(x) / Math.sqrt(2.0);

    const t = 1.0 / (1.0 + p * absX);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX);

    return 0.5 * (1.0 + sign * y);
}
