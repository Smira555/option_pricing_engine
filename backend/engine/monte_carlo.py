import numpy as np

def monte_carlo_price(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    option_type: str = "call", 
    num_simulations: int = 100000, 
    use_antithetic: bool = True
) -> dict:
    """
    Calculate the European option price using Monte Carlo simulation.
    Supports Antithetic Variates for variance reduction.
    """
    if T <= 0:
        payoff = max(S - K, 0.0) if option_type.lower() == "call" else max(K - S, 0.0)
        return {
            "price": payoff,
            "std_err": 0.0,
            "ci_lower": payoff,
            "ci_upper": payoff,
            "paths": [],
            "convergence": []
        }

    # Set random seed for reproducibility
    np.random.seed(42)

    # For Antithetic Variates, we generate half the random numbers and mirror them
    if use_antithetic:
        half_sims = num_simulations // 2
        # Generate standard normal random numbers
        Z = np.random.standard_normal(half_sims)
        # Combine Z and -Z (negatively correlated)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = np.random.standard_normal(num_simulations)

    # Simulate stock price at maturity T
    # S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z_full
    S_T = S * np.exp(drift + diffusion)

    # Calculate payoff at maturity
    if option_type.lower() == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    # Discount payoffs back to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = payoffs * discount_factor

    # Calculate price and statistical properties
    price = np.mean(discounted_payoffs)
    
    # Calculate standard error of the estimator
    if use_antithetic:
        # For antithetic, we pair payoff[i] and payoff[i + half_sims]
        paired_payoffs = 0.5 * (discounted_payoffs[:half_sims] + discounted_payoffs[half_sims:])
        std_err = np.std(paired_payoffs) / np.sqrt(half_sims)
    else:
        std_err = np.std(discounted_payoffs) / np.sqrt(num_simulations)

    # 95% Confidence Interval bounds (z = 1.96)
    ci_lower = price - 1.96 * std_err
    ci_upper = price + 1.96 * std_err

    # Generate sample paths for visualization (e.g., first 10 paths)
    # We will simulate 10 paths with 50 time steps each
    num_steps = 50
    dt = T / num_steps
    t = np.linspace(0, T, num_steps + 1)
    
    sample_paths = []
    for _ in range(10):
        path = [S]
        for _ in range(num_steps):
            z = np.random.standard_normal()
            next_S = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
            path.append(float(next_S))
        sample_paths.append(path)

    # Calculate convergence curve (running average at standard checkpoints)
    # Checkpoints: 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000 (up to num_simulations)
    checkpoints = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, num_simulations]
    checkpoints = [cp for cp in checkpoints if cp <= num_simulations]
    
    convergence = []
    for cp in checkpoints:
        if use_antithetic:
            # For checkpoint cp, make sure it's even to preserve pairs
            cp_half = cp // 2
            if cp_half == 0:
                cp_half = 1
            cp_actual = cp_half * 2
            mean_val = np.mean(discounted_payoffs[:cp_actual])
        else:
            mean_val = np.mean(discounted_payoffs[:cp])
        convergence.append({"paths": cp, "price": float(mean_val)})

    return {
        "price": float(price),
        "std_err": float(std_err),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "paths": sample_paths,
        "time_grid": t.tolist(),
        "convergence": convergence
    }
