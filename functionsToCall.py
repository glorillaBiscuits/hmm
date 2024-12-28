import time
import csv
import math
from math import sqrt
from scipy.stats import skewnorm, skew, kurtosis
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor
from metrics import portfolioMetrics  # Assuming you have a module for portfolio metrics

# Parallelized file processing
def process_file(file_path, asset_name, dividend_yield):
    try:
        prices = []
        log_returns = []
        daily_yield = dividend_yield / 252  # Convert yearly dividend to daily yield

        with open(file_path, 'r') as rf:
            reader = csv.reader(rf, delimiter=',')
            next(reader)  # Skip the header row

            for row in reader:
                try:
                    prices.append(float(row[1]))  # Assume the second column contains closing prices
                except (ValueError, IndexError):
                    continue  # Skip rows with invalid data

        if not prices:
            print(f"File '{file_path}' has no valid price data. Skipping.")
            return asset_name, None

        if len(prices) > 1:
            for j in range(1, len(prices)):
                try:
                    adjusted_price_t1 = prices[j] + (prices[j] * daily_yield)
                    log_return = math.log(adjusted_price_t1 / prices[j - 1])  # Adjusted log return
                    log_returns.append(log_return)
                except (ValueError, ZeroDivisionError):
                    log_returns.append(0)

        return asset_name, {'prices': prices, 'log_returns': log_returns}
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Skipping.")
    except Exception as e:
        print(f"Unexpected error while processing '{file_path}': {e}")
    return asset_name, None

def csvRead(csvDict, yearlyDividends):
    data_dict = {}

    with ProcessPoolExecutor() as executor:
        results = executor.map(
            process_file,
            csvDict['files'],
            csvDict['names'],
            yearlyDividends
        )

    for asset_name, result in results:
        if result is not None:
            data_dict[asset_name] = result

    return data_dict

# Align arrays of log returns
def align_arrays_same_length(data_dict):
    min_length = min(len(data_dict[name]['log_returns']) for name in data_dict)
    truncated_data = {name: data_dict[name]['log_returns'][:min_length] for name in data_dict}
    return pd.DataFrame(truncated_data)

# Half Kelly Criterion
def halfKellyCriterion(data_dict):
    df = align_arrays_same_length(data_dict)
    risk_free_rate = 0.04 / 252  # Example: 4% annual risk-free rate converted to daily
    expected_returns = df.mean().values - risk_free_rate
    cov_matrix = df.cov().values

    if np.linalg.det(cov_matrix) == 0:
        print("Covariance matrix is singular.")
        return pd.DataFrame()

    inv_cov_matrix = np.linalg.inv(cov_matrix)
    kelly_weights = np.dot(inv_cov_matrix, expected_returns) / 2
    assets = df.columns
    results = pd.DataFrame({'Asset': assets, 'Half Kelly Weight': kelly_weights})
    results = results[results['Half Kelly Weight'] > 0]

    if results['Half Kelly Weight'].sum() < 1:
        bond_weight = 1 - results['Half Kelly Weight'].sum()
        bond_entry = pd.DataFrame({'Asset': ['Bonds'], 'Half Kelly Weight': [bond_weight]})
        results = pd.concat([results, bond_entry], ignore_index=True)
    else:
        results['Half Kelly Weight'] /= results['Half Kelly Weight'].sum()

    return results

# Equal Weights
def equalWeightsDF(data_dict):
    assets = list(data_dict.keys())
    equal_weight = 1 / len(assets) if assets else 0
    return pd.DataFrame({'Asset': assets, 'Weight': [equal_weight] * len(assets)})

# GBM Price Simulation
def gbmPrice_tf(S0, mu, sigma, years, dt, n_paths, convergence_threshold=1e-5):
    mu = mu * 252
    sigma = sigma * tf.sqrt(252.0)
    n_steps = years/ dt
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * tf.sqrt(dt)

    rand = tf.random.normal(
        shape=(tf.cast(n_paths, tf.int32), tf.cast(n_steps, tf.int32)),
        mean=0.0,
        stddev=1.0
    )
    log_returns = drift + diffusion * rand
    cumulative_returns = tf.cumsum(log_returns, axis=1)
    final_prices = S0 * tf.exp(cumulative_returns[:, -1])

    mean_final = tf.reduce_mean(final_prices)
    std_final = tf.math.reduce_std(final_prices)

    if std_final / mean_final < convergence_threshold:
        return mean_final.numpy()

    return mean_final.numpy()

# Historical Returns Simulation
def historicalReturnsSample_tf(data_dict, names, years, dt, S0, n_simulations, convergence_threshold=1e-4):
    penaltyMetrics = {}
    daily_sim_price_dict = {}
    n_steps = int(years / dt)

    S0 = tf.convert_to_tensor(S0, dtype=tf.float32)
    fitted_params = {name: skewnorm.fit(data_dict[name]['log_returns']) for name in names}

    @tf.function
    def simulate_all_paths():
        sampled_returns = tf.stack([
            tf.random.stateless_normal(
                shape=(n_simulations, n_steps),
                mean=fitted_params[name][1],
                stddev=fitted_params[name][2],
                seed=(42, idx)
            )
            for idx, name in enumerate(names)
        ])
        return sampled_returns

    log_returns = simulate_all_paths()
    cumulative_returns = tf.cumsum(log_returns, axis=2)
    daily_prices = tf.exp(cumulative_returns) * tf.expand_dims(tf.expand_dims(S0, axis=1), axis=2)

    for idx, name in enumerate(names):
        final_prices = daily_prices[idx, :, -1]
        mean_final_price = tf.reduce_mean(final_prices)
        std_final_price = tf.math.reduce_std(final_prices)

        penaltyMetrics[name] = {
            "skewness": skew(data_dict[name]['log_returns']),
            "kurtosis": kurtosis(data_dict[name]['log_returns']),
        }

        relative_change = std_final_price / mean_final_price
        if relative_change < convergence_threshold:
            print(f"{name}: Converged after {n_simulations} simulations")

        mean_daily_prices = tf.reduce_mean(daily_prices[idx], axis=0).numpy()
        daily_sim_price_dict[name] = mean_daily_prices.tolist()
        data_dict[name]['Simulated Prices ' + name] = mean_daily_prices.tolist()

    return penaltyMetrics, daily_sim_price_dict

def portfolio_objective(
    weights,
    mean_returns,
    cov_matrix,
    penaltyMetrics,
    lambda1,
    lambda2,
    lambda3,
    names,
    equalDF,
    bond_return=0.000016,  # Approximate daily return for 4% yearly return
    bond_volatility=1e-6,  # Negligible volatility
    bond_skewness=0.0,     # Default skewness
    bond_kurtosis=3.0      # Default kurtosis (normal distribution)
):
    """
    Objective function to optimize the portfolio weights, including dynamically adding bonds.
    """
    bond_weight = weights[-1]  # Last weight corresponds to bonds

    # Portfolio return
    portfolio_return = np.dot(weights[:-1], mean_returns) + bond_weight * bond_return

    # Portfolio volatility
    portfolio_volatility = np.sqrt(
        np.dot(weights[:-1].T, np.dot(cov_matrix, weights[:-1])) + bond_weight ** 2 * bond_volatility ** 2
    )

    # Portfolio kurtosis and skewness
    asset_kurtosis = np.array([penaltyMetrics[name]["kurtosis"] for name in names])
    asset_skewness = np.array([penaltyMetrics[name]["skewness"] for name in names])
    portfolio_kurtosis = np.sum(weights[:-1] ** 2 * asset_kurtosis) + bond_weight ** 2 * bond_kurtosis
    portfolio_skewness = np.sum(weights[:-1] ** 2 * asset_skewness) + bond_weight ** 2 * bond_skewness

    # Objective function
    objective = (
        portfolio_return
        - lambda1 * portfolio_volatility
        - lambda2 * portfolio_kurtosis
        + lambda3 * portfolio_skewness
    )

    return -objective  # Minimize (negative because we want to maximize)

def weightOptimize(data_dict, years, dt, equalDF, n_simulations):
    """
    Optimizes portfolio weights to maximize the objective function.

    Args:
    - data_dict (dict): Historical data dictionary with asset names as keys.
    - T (float): Total simulation time (in years).
    - dt (float): Time step size (fraction of a year).
    - equalDF (pd.DataFrame): DataFrame containing equal weights.
    - n_simulations (int): Number of simulations.

    Returns:
    - np.ndarray: Optimized portfolio weights.
    """
    # Weighting factors for penalties
    lambda1, lambda2, lambda3 = 0.5, 0.3, 0.2

    # Initialize weights from equal weights DataFrame
    initial_weights = {name: equalDF.loc[equalDF['Asset'] == name, 'Weight'].values[0] for name in data_dict.keys()}

    # Bond characteristics
    bond_return = 0.000016  # Approximate daily return for 4% yearly return
    bond_volatility = 1e-6  # Negligible volatility
    bond_skewness = 0.0     # Default skewness
    bond_kurtosis = 3.0     # Default kurtosis (normal distribution)

    # Extract asset names and prices
    names = list(data_dict.keys())
    S0 = np.array([data_dict[name]['prices'][-1] for name in names], dtype=float)

    # Simulate prices and compute penalty metrics
    penaltyMetrics, simulated_prices = historicalReturnsSample_tf(data_dict, names, years, dt, S0, n_simulations)

    # Align log returns for covariance and mean calculations
    df = align_arrays_same_length(data_dict)
    mean_returns = df.mean().values
    cov_matrix = df.cov().values

    # Define constraints and bounds
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # Weights must sum to 1
    bounds = [(0, 1) for _ in range(len(names) + 1)]  # Bounds for assets and bonds

    # Initialize weights (add bonds)
    weights = [initial_weights[name] for name in names]
    weights.append(1 - sum(weights))  # Bond weight ensures total = 1

    # Optimization
    result = minimize(
        portfolio_objective,
        weights,
        args=(mean_returns,
              cov_matrix,
              penaltyMetrics,
              lambda1,
              lambda2,
              lambda3,
              names,
              equalDF,
              bond_return,
              bond_volatility,
              bond_skewness,
              bond_kurtosis),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    # Debugging: Check result
    if not result.success:
        print("Optimization failed:", result.message)

    altNames = names
    altNames.append("Bonds")
    threshold = 9e-3  # Define a threshold for filtering
    data_frame = pd.DataFrame({'Asset': altNames , 'Weight': result.x})
    data_frame = data_frame.sort_values(by=['Weight'], ascending=False)
    data_frame = data_frame[data_frame['Weight'] > threshold]
    data_frame['Weight'] = data_frame['Weight'].round(2)
    data_frame['Weight'] /= data_frame['Weight'].sum()

    return data_frame, simulated_prices

def optimization(initialPortfolio, data_dict, dt, outerSim, maxVolatility, n_paths, n_simulations, T):
    optimizedPortfolio = {}
    names = initialPortfolio['Asset'].tolist()
    initial_weights = dict(zip(names, initialPortfolio['Weight']))
    lambda1, lambda2, lambda3 = 0.5, 0.3, 0.2

    # Portfolio metrics
    noLogSum = sum(initial_weights[name] * data_dict[name]['prices'][-1] for name in names)
    optimizedPortfolio['originalSum'] = noLogSum

    # Simulated prices and penalty metrics
    S0 = [data_dict[name]['prices'][-1] for name in names]
    penaltyMetrics, simulated_prices = historicalReturnsSample_tf(data_dict, names, 5, dt, S0, n_simulations)

    # Align asset returns for covariance calculation
    for name in names:
        simLog = np.diff(np.log(simulated_prices[name]))
        data_dict[name]['log_returns'].extend(simLog)
    min_length = min(len(data_dict[name]['log_returns']) for name in names)
    aligned_log_returns = np.array([data_dict[name]['log_returns'][:min_length] for name in names]).T
    cov_matrix = np.cov(aligned_log_returns, rowvar=False)

    original_weights_vector = np.array(list(initial_weights.values()))
    original_volatility = np.sqrt(original_weights_vector.T @ cov_matrix @ original_weights_vector)
    optimizedPortfolio['originalVolatility'] = original_volatility  # Daily volatility
    for name, s0 in zip(names, S0):  # Synchronize iteration using zip
        mu = np.mean(data_dict[name]['log_returns'])
        sigma = np.std(data_dict[name]['log_returns'])
        optimizedPortfolio[name + 'GbmEndPrice'] = gbmPrice_tf(s0, float(mu), float(sigma), tf.constant(5, dtype=tf.float32), 1 / 252,
                                                               n_paths)
    # Monte Carlo optimization parameters
    bestObjective = -float("inf")
    optimizedPortfolioWeights = {}
    step_history = []

    for simNumber in range(outerSim):
        perturbation_factor = max(0.001, 0.05 * (1 - simNumber / outerSim))

        # Perturb weights
        perturbed_weights = {
            name: max(0, weight + np.random.normal(0, perturbation_factor))
            for name, weight in initial_weights.items()
        }
        weight_sum = sum(perturbed_weights.values())
        perturbed_weights = {name: weight / weight_sum for name, weight in perturbed_weights.items()}

        # Calculate portfolio metrics
        portfolio_return = sum(perturbed_weights[name] * simulated_prices[name][-1] for name in names)
        portfolio_volatility = np.sqrt(
            np.dot(
                np.array(list(perturbed_weights.values())).T,
                np.dot(cov_matrix, np.array(list(perturbed_weights.values())))
            )
        )
        portfolio_kurtosis = sum(perturbed_weights[name] ** 2 * penaltyMetrics[name]["kurtosis"] for name in names)
        portfolio_skewness = sum(perturbed_weights[name] ** 2 * penaltyMetrics[name]["skewness"] for name in names)

        # Scale with bonds
        scaled_weights = adjust_bond_allocation(
            optimized_weights=perturbed_weights,
            risky_volatility=portfolio_volatility,
            max_volatility=maxVolatility
        )

        # Recalculate scaled metrics
        risky_weights_vector = np.array([scaled_weights[name] for name in perturbed_weights.keys()])
        bond_weight = scaled_weights["Bonds"]
        scaled_variance = (
                np.dot(risky_weights_vector.T, np.dot(cov_matrix, risky_weights_vector))
                + bond_weight ** 2 * 1e-6  # Bond variance
        )
        scaled_volatility = np.sqrt(scaled_variance)

        # Compute scaled objective
        scaledObjective = (
                portfolio_return + bond_weight * (0.04 * T)  # Daily bond returns
                - lambda1 * scaled_volatility
                - lambda2 * portfolio_kurtosis
                + lambda3 * portfolio_skewness
        )

        # Update best portfolio
        if scaledObjective > bestObjective:
            bestObjective = scaledObjective
            optimizedPortfolioWeights = {name: round(weight, 4) for name, weight in scaled_weights.items() if
                                         name != "Bonds"}
            optimizedPortfolio["volatility"] = scaled_volatility * np.sqrt(252)
            for name in names:
                optimizedPortfolio[name + 'EndPrice'] = simulated_prices[name][-1]

    # Final portfolio adjustments
    optimizedPortfolio['optimizedPortfolioWeights'] = adjust_bond_allocation(
        optimizedPortfolioWeights, optimizedPortfolio["volatility"], maxVolatility
    )
    optimizedPortfolio['totalReturns'] = sum(
        optimizedPortfolio['optimizedPortfolioWeights'][name] * optimizedPortfolio[name + 'EndPrice']
        for name in names
    )

    # Final Portfolio Weights
    final_weights_vector = np.array([optimizedPortfolio['optimizedPortfolioWeights'][name] for name in names])

    # Calculate Portfolio Volatility
    final_volatility = np.sqrt(
        np.dot(final_weights_vector.T, np.dot(cov_matrix, final_weights_vector))
    )

    # Adjust Volatility to Annualized
    final_annualized_volatility = final_volatility * np.sqrt(252)

    # Store in Optimized Portfolio
    optimizedPortfolio['volatility'] = final_annualized_volatility

    return optimizedPortfolio

def adjust_bond_allocation(optimized_weights, risky_volatility, max_volatility):
    """
    Adjusts the portfolio allocation between risky assets and bonds to meet max volatility.

    Args:
    - optimized_weights (dict): Optimized weights for risky assets.
    - risky_volatility (float): Volatility of the optimized risky portfolio.
    - bond_return (float): Return of the bonds (daily return in this context).
    - bond_volatility (float): Volatility of the bonds (daily volatility in this context).
    - max_volatility (float): Maximum allowable portfolio volatility (annualized).

    Returns:
    - dict: Final portfolio weights including bonds.
    """
    bond_return = 0.04 / 252  # Daily bond return
    bond_volatility = np.sqrt(1e-6)  # Daily bond volatility

    # Check if risky volatility is already within bounds
    if risky_volatility <= max_volatility:
        # No adjustment necessary; use current weights and set bond allocation to 0
        final_weights = optimized_weights.copy()
        final_weights["Bonds"] = 0
        return final_weights

    def portfolio_volatility(alpha):
        """
        Calculate the portfolio's volatility as a function of the allocation to risky assets (alpha).
        """
        combined_variance = (alpha ** 2 * risky_volatility ** 2) + ((1 - alpha) ** 2 * bond_volatility ** 2)
        return np.sqrt(combined_variance)

    # Objective: Minimize alpha while keeping portfolio volatility below max_volatility
    def objective(alpha):
        # Maximize risky allocation (invert for minimize)
        return -alpha

    # Constraint: Portfolio volatility must be <= daily_max_volatility
    constraints = [{"type": "ineq", "fun": lambda alpha: max_volatility - portfolio_volatility(alpha)}]

    # Bounds for alpha: Between 0 (all bonds) and 1 (all risky assets)
    bounds = [(0 + 1e-6, 1 - 1e-6)]  # Add small epsilons to avoid precision issues

    # Initial guess: Start with a safe allocation based on volatility ratio
    initial_alpha = min(1, max_volatility / risky_volatility)

    # Optimize
    result = minimize(
        objective,
        initial_alpha,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP"  # Sequential Least Squares Programming
    )

    if not result.success:
        # Add fallback or debugging if needed
        raise ValueError(f"Bond allocation optimization failed: {result.message}")

    # Optimal alpha
    alpha = result.x[0]

    # Scale risky and bond weights
    risky_weights_scaled = {name: weight * alpha for name, weight in optimized_weights.items()}
    bond_weight = 1 - alpha

    # Combine weights
    final_weights = risky_weights_scaled
    final_weights["Bonds"] = bond_weight

    return final_weights


# Main Execution
if __name__ == "__main__":
    filesCsv = ['apple.csv', 'ko.csv', 'jnj.csv']
    dividends = [0.0, 0.0, 0.0]

    csvDict = {
        'names': [file.replace('.csv', '') for file in filesCsv],
        'files': filesCsv
    }

    n_paths = 10000
    n_simulations = 100000
    outerSim = 5
    maxVolatility = 0.15
    T = 5
    years = 1

    start_time = time.time()
    data_dict = csvRead(csvDict, dividends)
    dt = 1 / 252

    results = halfKellyCriterion(data_dict)
    #print(results)
    initialPortfolio = equalWeightsDF(data_dict)
    opt_weights = weightOptimize(data_dict, years, dt, initialPortfolio, n_simulations)
    #print(opt_weights[0])
    optimizedPortfolio = optimization(initialPortfolio, data_dict, dt, outerSim, maxVolatility, n_paths, n_simulations, T)
    portfolioMetrics(optimizedPortfolio, data_dict, 5)
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds")


    def generate_innovations(best_distribution, best_params, size, device):
        """
        Generate innovations (epsilon_t) dynamically from the best-fit distribution.

        Args:
        - best_distribution (str): Name of the distribution.
        - best_params (dict): Parameters of the distribution.
        - size (int): Number of innovations to generate.
        - device: PyTorch device ('mps' or 'cpu').

        Returns:
        - torch.Tensor: Generated innovations.
        """
        try:
            dist = getattr(distributions, best_distribution)
        except AttributeError:
            raise ValueError(f"Unsupported distribution: {best_distribution}")

        # Generate random variates
        innovations = dist.rvs(size=size, **best_params)
        return torch.tensor(innovations, dtype=torch.float32, device=device)


    # Normalize log returns

    def normalize_log_returns(log_returns):
        scaler = StandardScaler()
        log_returns_norm = scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
        return log_returns_norm, scaler


    def reverse_normalization(transformed_returns, scaler):
        return scaler.inverse_transform(transformed_returns.reshape(-1, 1)).flatten()


    def initialize_garch_params(log_returns):
        """
        Initialize GARCH parameters using the unconditional variance.

        Args:
        - log_returns: Historical log returns.

        Returns:
        - dict: Initial GARCH parameters.
        """
        var_returns = np.var(log_returns)
        alpha, beta = 0.1, 0.8  # Reasonable starting values
        omega = var_returns * (1 - alpha - beta)
        return {"omega": omega, "alpha": alpha, "beta": beta}


    def estimate_garch_mle(log_returns, window_size=5):
        """
        Estimate GARCH(1,1) parameters using a regression-based approach.

        Args:
            log_returns (np.array): Array of log returns.
            window_size (int): Rolling window size for variance estimation.

        Returns:
            dict: Estimated GARCH(1,1) parameters (omega, alpha, beta).
        """
        returns_arrays = np.array(log_returns)
        rolling_variance = pd.Series(returns_arrays ** 2).rolling(window=window_size).mean().dropna()
        rolling_variance = rolling_variance.values

        y = rolling_variance[1:]
        lagged_returns_squared = returns_arrays[window_size - 1: -1] ** 2
        lagged_variance = rolling_variance[:-1]
        X = np.column_stack((np.ones(len(lagged_returns_squared)), lagged_returns_squared, lagged_variance))

        model = LinearRegression()
        model.fit(X, y)

        omega = model.coef_[0]
        alpha = model.coef_[1]
        beta = model.coef_[2]

        r_squared = model.score(X, y)
        print(f"R^2: {r_squared:.4f}")
        print(f"Mean Squared Error: {mean_squared_error(y, model.predict(X)):.4e}")

        return {"omega": omega, "alpha": alpha, "beta": beta}


    def custom_garch_tf(log_returns, n_steps, omega, alpha, beta, device='cpu'):
        """
        Custom GARCH(1,1) model simulation.

        Args:
        - log_returns: Historical log returns.
        - n_steps: Number of steps to simulate.
        - omega: Constant term in the GARCH model.
        - alpha: Coefficient for past squared errors.
        - beta: Coefficient for past volatility.
        - device: PyTorch device for computation ('mps' or 'cpu').

        Returns:
        - torch.Tensor: Simulated returns.
        - torch.Tensor: Simulated volatilities.
        """
        device = torch.device(device)

        log_returns_norm, scaler = normalize_log_returns(log_returns)
        best_distribution, best_params = find_best_distributions(log_returns_norm)

        volatilities = torch.zeros(n_steps, dtype=torch.float32, device=device)
        returns = torch.zeros(n_steps, dtype=torch.float32, device=device)
        volatilities[0] = torch.sqrt(torch.tensor(omega / (1 - alpha - beta), dtype=torch.float32, device=device))
        innovations = generate_innovations(best_distribution, best_params, size=n_steps, device=device)

        for t in range(1, n_steps):
            volatilities[t] = torch.sqrt(omega + alpha * returns[t - 1] ** 2 + beta * volatilities[t - 1] ** 2)
            returns[t] = volatilities[t] * innovations[t]

        returns_original = reverse_normalization(returns.cpu().numpy(), scaler)
        return torch.tensor(returns_original, dtype=torch.float32, device=device), volatilities


    def simulate_garch_paths_custom(log_returns, T, dt, S0, params, n_simulations, device=None):
        """
        Simulates GARCH(1,1) paths, applying scaling for model fitting and reversing scaling for final output.

        Args:
            log_returns (np.array): Historical log returns for the asset.
            T (float): Total simulation time in years.
            dt (float): Time step size as a fraction of a year.
            S0 (float): Initial price of the asset.
            params (dict): Estimated GARCH parameters (omega, alpha, beta).
            n_simulations (int): Number of simulation paths.
            device (str): Computation device.

        Returns:
            np.array: Mean daily prices across simulations.
        """
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        n_steps = int(T / dt)
        log_returns_scaled = np.array(log_returns, dtype=np.float32) * 100
        omega = params['omega']
        alpha = params['alpha']
        beta = params['beta']

        returns, _ = custom_garch_tf(log_returns_scaled, n_steps, omega, alpha, beta, device=device)
        returns /= 100
        returns = returns.unsqueeze(0).repeat(n_simulations, 1)

        cumulative_returns = torch.cumsum(returns, dim=1)
        cumulative_prices = S0 * torch.exp(cumulative_returns)
        mean_daily_prices = torch.mean(cumulative_prices, dim=0)

        return mean_daily_prices.cpu().numpy()
