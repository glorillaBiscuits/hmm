import time
import csv
import math
import pandas as pd



def dict(filesCsv, yearlyDividends, window, recent):
    """
    Processes data for backtesting, splitting into training and testing datasets.

    Args:
    - filesCsv: List of file names (CSV).
    - yearlyDividends: Dictionary of yearly dividends for each asset.
    - window: Fraction of data to use for testing (e.g., 0.2 for 20% testing).
    - recent: Number of years of recent data to use (e.g., 5 for the last 5 years).

    Returns:
    - train_data: Dictionary containing training data for each asset.
    - test_data: Dictionary containing testing data for each asset.
    - data_dict: Full dataset from csvRead.
    - T: Time horizon for simulations (in years, based on the testing window).
    - S0_dict: Starting prices for simulations.
    """
    csvDict = {}
    csvDict['names'] = [file.replace('.csv', '') for file in filesCsv]
    csvDict['files'] = filesCsv
    csvDict['numberOfAsset'] = len(filesCsv)

    # Read data
    data_dict = csvRead(csvDict, yearlyDividends)

    # Determine the minimum dataset length across all assets
    days_data = min(len(data_dict[name]['log_returns']) for name in data_dict)
    recent_years = min(recent * 252, days_data)

    # Truncate data to the most recent period
    recent_data = {
        name: {
            'prices': data_dict[name]['prices'][-recent_years:],
            'log_returns': data_dict[name]['log_returns'][-recent_years:]
        }
        for name in data_dict
    }

    # Calculate the testing window size
    day_window = int(recent_years * window)
    if recent_years <= day_window:
        raise ValueError("Insufficient data points for the specified testing window.")

    # Calculate the simulation horizon
    T = day_window / 252  # Convert days to years

    # Split into training and testing datasets
    train_data = {
        name: {
            'prices': recent_data[name]['prices'][:-day_window],
            'log_returns': recent_data[name]['log_returns'][:-day_window],
            'S0': recent_data[name]['prices'][-day_window - 1]
        }
        for name in recent_data
    }

    test_data = {
        name: {
            'prices': recent_data[name]['prices'][-day_window:],
            'log_returns': recent_data[name]['log_returns'][-day_window:],
            'final': recent_data[name]['prices'][-1]
        }
        for name in recent_data
    }

    # Create S0_dict for starting prices
    S0_dict = {name: train_data[name]['S0'] for name in train_data}

    return train_data, test_data, data_dict, T, S0_dict




def evaluate_metrics(real_log_returns, simulated_log_returns, name):
    metrics = {
        "Real Volatility": np.std(real_log_returns),
        "Simulated Volatility": np.std(simulated_log_returns),
        "Real Skewness": skew(real_log_returns),
        "Simulated Skewness": skew(simulated_log_returns),
        "Real Kurtosis": kurtosis(real_log_returns),
        "Simulated Kurtosis": kurtosis(simulated_log_returns),
    }
    print(name, "Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


def csvRead(csvDict, yearlyDividends):
    """
    Reads raw prices from CSV files and optionally calculates dividend-adjusted log returns.

    Parameters:
        csvDict (dict): A dictionary with:
            - 'files': List of CSV file paths (str).
            - 'names': List of corresponding asset names (str).
        yearlyDividends (list): List of yearly dividend yields (fractions) for each asset.

    Returns:
        dict: A single dictionary with dividend-adjusted prices and log returns for each asset.
    """
    data_dict = {}

    for i, (assetFile, assetName) in enumerate(zip(csvDict['files'], csvDict['names'])):
        try:
            with open(assetFile, 'r') as rf:  # Open each file
                reader = csv.reader(rf, delimiter=',')
                next(reader)  # Skip the header row

                prices = []  # List to hold raw prices for the current asset
                log_returns = []  # List to hold dividend-adjusted log returns if calculated
                daily_yield = yearlyDividends[i] / 252  # Convert yearly dividend to daily yield

                for row in reader:
                    try:
                        # Assume the second column contains closing prices
                        prices.append(float(row[1]))  # Convert to float
                    except (ValueError, IndexError):
                        continue  # Skip rows with invalid data

                if not prices:
                    print(f"File '{assetFile}' has no valid price data. Skipping.")
                    continue

                # If calculate_log_returns is True, compute log returns adjusted for dividends
                if  len(prices) > 1:
                    for j in range(1, len(prices)):
                        try:
                            adjusted_price_t1 = prices[j] + (prices[j] * daily_yield)  # Add dividend yield
                            log_return = math.log(adjusted_price_t1 / prices[j - 1])  # Adjusted log return
                            log_returns.append(log_return)
                        except (ValueError, ZeroDivisionError):
                            log_returns.append(0)  # Default to 0 for invalid log returns

                # Store prices and log returns in a single dictionary
                data_dict[assetName] = {
                    'prices': prices,
                    'log_returns': log_returns
                }

        except FileNotFoundError:
            print(f"File '{assetFile}' not found. Skipping.")
        except Exception as e:
            print(f"Unexpected error while processing '{assetFile}': {e}")

    return data_dict



def gbmPrice_parallel(S0, mu, sigma, T, dt, n_paths, max_workers=4):
    mu = mu * 252  # Annualize
    sigma = sigma * sqrt(1 / 252)  # Annualize
    n_steps = int(T / dt)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    args = [(S0, drift, diffusion, n_steps) for _ in range(n_paths)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        final_prices = list(executor.map(simulate_path, args))

    return np.mean(final_prices)

def simulate_path(args):
    S0, drift, diffusion, n_steps = args
    rand = np.random.normal(0, 1, n_steps)
    log_returns = drift + diffusion * rand
    cumulative_returns = np.cumsum(log_returns)
    return S0 * np.exp(cumulative_returns[-1])


from concurrent.futures import ProcessPoolExecutor



def gbm_parallel_converge(S0, mu, sigma, T, dt, n_paths, max_workers=4, convergence_threshold=1e-6, max_batches=400):
    """
    Simulates GBM paths in parallel and checks for convergence.

    Parameters:
        S0 (float): Initial stock price.
        mu (float): Drift term (annualized return).
        sigma (float): Volatility (annualized).
        T (float): Time horizon in years.
        dt (float): Time step size in years.
        n_paths (int): Total number of paths to simulate.
        max_workers (int): Maximum number of parallel workers.
        convergence_threshold (float): Threshold for convergence.
        max_batches (int): Maximum number of batches for convergence check.

    Returns:
        tuple: (final_mean, batches_processed, converged)
            - final_mean (float): Mean of final prices after convergence.
            - batches_processed (int): Number of batches processed.
            - converged (bool): Whether convergence was achieved.
    """
    mu = mu * 252  # Annualize
    sigma = sigma * np.sqrt(252)  # Annualize
    n_steps = int(T / dt)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Split paths into batches for convergence checking
    n_paths_per_batch = n_paths // max_batches
    args = [(S0, drift, diffusion, n_steps) for _ in range(n_paths_per_batch)]

    running_mean, prev_mean = 0.0, None
    total_paths = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch in range(max_batches):
            final_prices = list(executor.map(simulate_converge, args))
            batch_mean = np.mean(final_prices)

            # Update running mean with incremental average
            total_paths += n_paths_per_batch
            running_mean += (batch_mean - running_mean) / (batch + 1)

            # Convergence check
            if prev_mean is not None:
                relative_change = abs(running_mean - prev_mean) / (prev_mean + 1e-10)
                if relative_change < convergence_threshold:
                    print(f"Converged after {batch + 1} batches. Final mean: {running_mean}")
                    return running_mean, batch + 1, True

            prev_mean = running_mean

    print(f"Reached max batches without convergence. Final mean: {running_mean}")
    return running_mean, max_batches, False


def simulate_converge(args):
    S0, drift, diffusion, n_steps = args
    rand = np.random.normal(0, 1, n_steps)
    log_returns = drift + diffusion * rand
    cumulative_returns = np.cumsum(log_returns)
    return S0 * np.exp(cumulative_returns[-1])

import tensorflow as tf
from scipy.stats import skew, kurtosis, gaussian_kde



def historicalReturnsSample_t(data_dict, names, T, dt, S0, n_simulations, history_length=50, convergence_threshold=1e-6):
    """
    Simulates historical returns for assets and computes penalty metrics with enhanced precision.

    Args:
    - data_dict (dict): Dictionary containing log returns for each asset.
    - names (list): List of asset names.
    - T (float): Total simulation time in years.
    - dt (float): Time step size as a fraction of a year.
    - S0 (list): Initial prices for each asset.
    - n_simulations (int): Number of simulations to run.
    - history_length (int): Number of steps to consider for convergence check.
    - convergence_threshold (float): Threshold for convergence.

    Returns:
    - dict: Penalty metrics (kurtosis and skewness) for each asset.
    - dict: Average daily prices across all simulations for each asset.
    """
    n_steps = int(T / dt)
    S0 = tf.convert_to_tensor(S0, dtype=tf.float32)  # Ensure S0 is a TensorFlow tensor
    penaltyMetrics = {}
    daily_sim_price_dict = {}

    for idx, name in enumerate(names):
        # Calculate skewness and kurtosis for the stock
        log_returns = np.array(data_dict[name]['log_returns'], dtype=np.float32)
        penaltyMetrics[name] = {
            "skewness": skew(log_returns),
            "kurtosis": kurtosis(log_returns),
        }
        data_dict[name]['Penalty Metrics ' + name] = penaltyMetrics[name]

        # Fit Gaussian Kernel Density Estimate (KDE) for log-returns
        kde = gaussian_kde(log_returns)

        # Initialize variables for simulation and convergence
        step_history = []
        all_simulated_prices = []

        for _ in range(n_simulations):
            # Sample returns using KDE for better tail behavior modeling
            sampled_returns = kde.resample(n_steps)[0]

            # Calculate cumulative returns and simulate daily prices
            cumulative_returns = tf.cast(tf.math.cumsum(sampled_returns), dtype=tf.float32)
            daily_prices = S0[idx] * tf.exp(cumulative_returns)
            all_simulated_prices.append(daily_prices.numpy())

            # Convergence Check
            step_history.append(daily_prices[-1].numpy())
            if len(step_history) > history_length:
                step_history.pop(0)

            if len(step_history) == history_length:
                recent_mean = np.mean(step_history)
                initial_mean = step_history[0]
                relative_change = abs(recent_mean - initial_mean) / (initial_mean + 1e-10)
                if relative_change < convergence_threshold:
                    break

        # Calculate mean daily prices across all simulations
        mean_daily_prices = np.mean(all_simulated_prices, axis=0).tolist()

        # Store results
        daily_sim_price_dict[name] = mean_daily_prices
        data_dict[name]['Simulated Prices ' + name] = mean_daily_prices

    return penaltyMetrics, daily_sim_price_dict



"""
Aligns arrays to the same minimum length using a DataFrame.

Parameters:
data_dict: Dictionary with arrays as values.
csvDict: Dictionary containing a 'names' key with a list of names corresponding to data_dict keys.

Returns:
A Pandas DataFrame with aligned arrays of the same length.
"""

from arch import arch_model

from scipy.stats import skew, kurtosis

def hrs(data_dict, names, T, dt, S0, n_simulations, history_length=50, convergence_threshold=1e-6):
    """
    Simulates historical returns using a GARCH(1,1) model for assets and computes penalty metrics.

    Args:
    - data_dict (dict): Dictionary containing log returns for each asset.
    - names (list): List of asset names.
    - T (float): Total simulation time in years.
    - dt (float): Time step size as a fraction of a year.
    - S0 (list): Initial prices for each asset.
    - n_simulations (int): Number of simulations to run.
    - history_length (int): Number of steps to consider for convergence check.
    - convergence_threshold (float): Threshold for convergence.

    Returns:
    - dict: Penalty metrics (kurtosis and skewness) for each asset.
    - dict: Average daily prices across all simulations for each asset.
    """
    n_steps = int(T / dt)
    S0 = np.array(S0, dtype=float)  # Ensure S0 is a numpy array for element-wise operations
    penaltyMetrics = {}
    daily_sim_price_dict = {}

    for idx, name in enumerate(names):
        # Retrieve historical log returns
        log_returns = np.array(data_dict[name]['log_returns'], dtype=np.float32)

        # Rescale log returns
        scale_factor = 100
        scaled_returns = log_returns * scale_factor

        # Calculate skewness and kurtosis
        penaltyMetrics[name] = {
            "skewness": skew(log_returns),
            "kurtosis": kurtosis(log_returns),
        }
        data_dict[name]['Penalty Metrics ' + name] = penaltyMetrics[name]

        # Fit GARCH(1,1) model to the rescaled log returns
        garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
        garch_fit = garch_model.fit(disp="off")

        # Initialize variables for simulation and convergence
        step_history = []
        all_simulated_prices = []

        for _ in range(n_simulations):
            # Simulate GARCH returns (ensure these are simple returns)
            simulated_data = garch_model.simulate(params=garch_fit.params, nobs=n_steps)

            # Convert simple returns to log-returns if needed
            simulated_returns = np.log(1 + simulated_data['data'] / scale_factor)

            # Calculate cumulative returns
            cumulative_returns = np.cumsum(simulated_returns)

            # Simulate daily prices using the cumulative log-returns
            daily_prices = S0[idx] * np.exp(cumulative_returns)

            # Store all simulated daily prices
            all_simulated_prices.append(daily_prices)

            # Convergence Check
            step_history.append(daily_prices.iloc[-1])
            if len(step_history) > history_length:
                step_history.pop(0)

            if len(step_history) == history_length:
                recent_mean = np.mean(step_history)
                initial_mean = step_history[0]
                relative_change = abs(recent_mean - initial_mean) / (initial_mean + 1e-10)
                if relative_change < convergence_threshold:
                    break

        """
        # Calculate mean daily prices across all simulations
        for index in range(len(all_simulated_prices)):
            print(all_simulated_prices[:10], '\n', all_simulated_prices[-10:])
        print(len(all_simulated_prices))
        """
        mean_daily_prices = np.mean(all_simulated_prices, axis=0).tolist()

        # Store results
        daily_sim_price_dict[name] = mean_daily_prices
        data_dict[name]['Simulated Prices ' + name] = mean_daily_prices

    return penaltyMetrics, daily_sim_price_dict



from arch import arch_model


def calculate_garch_volatility(log_returns, n_steps):
    """
    Fits a GARCH model to the log returns and forecasts future volatility.

    Args:
    - log_returns: Historical log returns (numpy array).
    - n_steps: Number of future steps to forecast volatility.

    Returns:
    - garch_volatility: Time-varying volatility for past and forecasted steps.
    """
    # Fit GARCH(1,1) model
    garch_model = arch_model(log_returns, vol='Garch', p=1, q=1, mean='Zero', rescale=True)
    garch_fit = garch_model.fit(disp="off")

    #evaluate_garch_fit(garch_fit)

    # Extract historical conditional volatility
    historical_volatility = garch_fit.conditional_volatility

    # Forecast future volatility
    garch_forecast = garch_fit.forecast(horizon=n_steps)
    forecast_volatility = np.sqrt(garch_forecast.variance.iloc[-1].values)  # Convert variance to volatility

    # Combine historical and forecasted volatility
    garch_volatility = np.concatenate((historical_volatility, forecast_volatility))



    return garch_volatility


def gbm_mean_garch_multiple(csvDict, data_dict, T, dt, n_paths):
    """
    Simulates GBM paths using GARCH-derived volatility for multiple assets and returns daily means.

    Args:
    - csvDict: Dictionary containing asset names and file information.
    - data_dict: Dictionary containing historical prices and log returns for each asset.
    - T: Total time horizon (in years).
    - dt: Time step size.
    - n_paths: Number of simulation paths.

    Returns:
    - result_dict: Dictionary where keys are asset names and values are lists of daily mean prices.
    """

    result_dict = {}

    for name in csvDict['names']:
        # Retrieve initial price and historical log returns excluding the last 252 days
        S0 = data_dict[name]['prices'][-1]
        log_returns = np.array(data_dict[name]['log_returns'][:-252], dtype=np.float32)
        mu = np.mean(log_returns) * 252  # Annualized mean return

        n_steps = int(T / dt)

        # Calculate GARCH volatility
        garch_volatility = calculate_garch_volatility(log_returns, n_steps)

        # Pre-allocate paths
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0

        for t in range(1, n_steps):
            # Use GARCH volatility dynamically
            sigma = garch_volatility[t % len(garch_volatility)]
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)

            # Simulate GBM paths
            z = np.random.normal(0, 1, size=n_paths)
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * z)

        # Calculate daily means across all paths
        daily_means = np.mean(paths, axis=0)

        # Store in result dictionary
        result_dict[name] = daily_means.tolist()

    return result_dict


from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot

def evaluate_garch_fit(garch_fit):
    """
    Evaluates the GARCH model fit using residual diagnostics.

    Args:
    - garch_fit: Fitted GARCH model (arch_model.fit).

    Returns:
    - None: Displays diagnostic plots and prints test results.
    """
    # Get residuals and standardized residuals
    residuals = garch_fit.resid
    standardized_residuals = garch_fit.resid / garch_fit.conditional_volatility

    # Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(standardized_residuals, lags=[10], return_df=True)
    print("Ljung-Box Test for Autocorrelation (p-values):")
    print(lb_test)

    # Plot residual diagnostics
    plt.figure(figsize=(12, 8))

    # Histogram of standardized residuals
    plt.subplot(2, 2, 1)
    sns.histplot(standardized_residuals, kde=True)
    plt.title("Histogram of Standardized Residuals")
    plt.xlabel("Standardized Residuals")

    # Q-Q plot of standardized residuals
    plt.subplot(2, 2, 2)
    probplot(standardized_residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Standardized Residuals")

    # Conditional volatility
    plt.subplot(2, 2, 3)
    plt.plot(garch_fit.conditional_volatility)
    plt.title("Conditional Volatility")
    plt.xlabel("Time")
    plt.ylabel("Volatility")

    # ACF of standardized residuals
    plt.subplot(2, 2, 4)
    pd.plotting.autocorrelation_plot(standardized_residuals)
    plt.title("ACF of Standardized Residuals")

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_real_vs_multiple_simulations(data_dict, simulation_dicts, asset_name):
    """
    Plots the last 252 values from real data and 252 simulated values from multiple simulations for a given asset.

    Args:
    - data_dict (dict): Dictionary containing real price data.
    - simulation_dicts (list): A list of dictionaries containing simulated prices.
                               Each dictionary should have the asset name as a key and the simulated prices as values.
    - asset_name (str): The name of the asset to plot.

    Returns:
    - None: Displays the plot.
    """
    # Extract the last 252 real prices
    real_prices = data_dict[asset_name]['prices'][-252:]

    # Create a plot
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label="Real Prices (Last 252 Days)", linewidth=2)

    # Iterate through simulation dictionaries and add to the plot
    for idx, sim_dict in enumerate(simulation_dicts):
        if asset_name in sim_dict:
            simulated = sim_dict[asset_name][:252]
            plt.plot(simulated, label=f"Simulated Prices {idx + 1} (252 Days)", linestyle='--', linewidth=1.5)

    # Add labels and title
    plt.title(f"Real vs Simulated Prices for {asset_name}", fontsize=14)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


def jump_diffusion_multiple(csvDict, shortData, S0, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump):
    """
    Simulates Jump Diffusion paths for multiple assets and calculates mean daily prices.

    Args:
    - csvDict: Dictionary containing asset names and file information.
    - shortLog: Dictionary containing log returns for each asset.
    - S0: List of initial prices for each asset.
    - T: Total time horizon (in years).
    - dt: Time step size (fraction of a year).
    - n_paths: Number of simulation paths.
    - lambda_jump: Jump intensity (expected jumps per year).
    - mu_jump: Mean of the jump size (log scale).
    - sigma_jump: Volatility of the jump size (log scale).

    Returns:
    - result_dict: Dictionary where keys are asset names and values are lists of mean daily prices.
    """
    result_dict = {}
    n_steps = int(T / dt)

    for idx, name in enumerate(csvDict['names']):
        # Extract initial price and log returns
        initial_price = S0[idx]
        log_returns = np.array(shortData[name]['log_returns'], dtype=np.float32)

        # Calculate drift (mu) and volatility (sigma) from log returns
        mu = np.mean(log_returns) * 252  # Annualized mean return
        sigma = np.std(log_returns) * np.sqrt(252)  # Annualized standard deviation

        # Pre-allocate paths
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = initial_price

        for t in range(1, n_steps):
            # Brownian motion component
            z = np.random.normal(0, 1, size=n_paths)
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * z

            # Poisson process for jump occurrence
            jumps = np.random.poisson(lambda_jump * dt, size=n_paths)  # 0 or 1 for most cases

            # Jump size (log-normal distributed)
            jump_magnitudes = np.exp(np.random.normal(mu_jump, sigma_jump, size=n_paths)) - 1

            # Update price paths
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion) * (1 + jumps * jump_magnitudes)

        # Calculate mean daily prices across all paths
        daily_means = np.mean(paths, axis=0)

        # Store in result dictionary
        result_dict[name] = daily_means.tolist()

    return result_dict


def simulate_jump_diffusion_paths(args):
    """
    Simulates Jump Diffusion paths with Bayesian-estimated parameters for a single asset.

    Args:
    - args: Tuple containing (name, S0, trace, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump).

    Returns:
    - name: Asset name.
    - daily_means: Mean of simulated paths for each day.
    - paths: Full simulated paths.
    """
    name, S0, trace, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump = args
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = S0

    for t in range(1, n_steps):
        # Sample drift and volatility from posterior distributions
        mu = np.random.choice(trace['mu'])
        sigma = np.random.choice(trace['sigma'])

        # Brownian motion component
        z = np.random.normal(0, 1, size=n_paths)
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * z

        # Poisson process for jumps
        jumps = np.random.poisson(lambda_jump * dt, size=n_paths)
        jump_magnitudes = np.exp(np.random.normal(mu_jump, sigma_jump, size=n_paths)) - 1

        # Update paths
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion) * (1 + jumps * jump_magnitudes)


    return name, paths


def jump_diffusion_paths(S0_dict, trace_dict, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump,
                                   n_processes=4):
    """
    Simulates Jump Diffusion paths for multiple assets and returns mean daily prices and full paths.

    Args:
    - S0_dict: Dictionary with asset names as keys and initial prices as values.
    - trace_dict: Dictionary with asset names as keys and MCMC traces as values.
    - T: Time horizon (years).
    - dt: Time step size.
    - n_paths: Number of paths.
    - lambda_jump: Jump intensity.
    - mu_jump: Mean jump size.
    - sigma_jump: Volatility of jump size.
    - n_processes: Number of parallel processes.

    Returns:
    - result_dict: Dictionary with asset names as keys and lists of mean daily prices as values.
    - full_path_dict: Dictionary with asset names as keys and full simulated paths as values.
    """
    # Prepare arguments for parallel processing
    args = [
        (name, S0_dict[name], trace_dict[name], T, dt, n_paths, lambda_jump, mu_jump, sigma_jump)
        for name in S0_dict
    ]

    # Parallelize simulation across assets
    with Pool(n_processes) as pool:
        results = pool.map(simulate_jump_diffusion_paths, args)

    # Collect results into dictionaries
    full_path_dict = {name: paths for name, paths in results}

    return full_path_dict


from scipy.stats import norm, halfnorm
import numpy as np

def bayesian_estimation_for_asset(args):
    name, log_returns = args
    try:
        mu_prior = norm(loc=np.mean(log_returns), scale=np.std(log_returns) / 2)
        sigma_prior = halfnorm(scale=np.std(log_returns))

        def log_likelihood(mu, sigma, data):
            sigma = max(sigma, 1e-6)
            return np.sum(norm.logpdf(data, loc=mu, scale=sigma))

        n_samples = min(10000, len(log_returns) * 10)
        mu_samples = mu_prior.rvs(size=n_samples)
        sigma_samples = sigma_prior.rvs(size=n_samples)

        log_posterior_probs = []
        for mu, sigma in zip(mu_samples, sigma_samples):
            if sigma <= 0:
                log_posterior_probs.append(-np.inf)
                continue
            log_prob = (
                log_likelihood(mu, sigma, log_returns) +
                mu_prior.logpdf(mu) +
                sigma_prior.logpdf(sigma)
            )
            log_posterior_probs.append(log_prob)

        log_posterior_probs = np.array(log_posterior_probs)
        max_log_prob = np.max(log_posterior_probs)
        posterior_probs = np.exp(log_posterior_probs - max_log_prob)
        if np.sum(posterior_probs) == 0:
            print(f"Log-likelihood range: {log_posterior_probs.min()} to {log_posterior_probs.max()}")
            raise ValueError("Posterior probabilities sum to zero.")
        posterior_probs /= np.sum(posterior_probs)

        indices = np.random.choice(range(n_samples), size=n_samples, p=posterior_probs)
        sampled_mu = mu_samples[indices]
        sampled_sigma = sigma_samples[indices] * np.sqrt(252)  # Annualize sigma

        print(f"{name}: Posterior Mu mean={np.mean(sampled_mu)}, Sigma mean={np.mean(sampled_sigma)}")
        trace = {"mu": sampled_mu, "sigma": sampled_sigma}
        return name, trace

    except Exception as e:
        print(f"Error processing {name}: {e}")
        return name, None


    except Exception as e:
        print(f"Error processing {name}: {e}")
        return name, None



def estimate_bayesian_parameters(shortData, n_processes=4):
    """
    Bayesian estimation of drift (mu) and volatility (sigma) for multiple assets using SciPy.

    Args:
    - shortData: Dictionary containing historical log returns for each asset.
    - n_processes: Number of parallel processes to use.

    Returns:
    - traceDict: Dictionary with asset names as keys and sampled posterior distributions for mu and sigma.
    """
    # Prepare arguments for parallel processing
    args = [(name, np.array(shortData[name]['log_returns'])) for name in shortData]

    # Parallelize Bayesian estimation
    if len(shortData) < n_processes:
        results = [bayesian_estimation_for_asset(arg) for arg in args]
    else:
        with Pool(n_processes) as pool:
            results = pool.map(bayesian_estimation_for_asset, args)

    # Combine results into a single dictionary
    traceDict = {}
    for entry in results:
        traceDict[entry[0]] = entry[1]

    return traceDict


from scipy.stats import gaussian_kde

def sample_with_kde(log_returns, n_samples, bandwidth=0.01):
    """
    Samples returns using Gaussian KDE with specified bandwidth.

    Args:
    - log_returns: Historical log returns.
    - n_samples: Number of samples to generate.
    - bandwidth: Bandwidth for KDE.

    Returns:
    - samples: Samples drawn from the KDE.
    """
    kde = gaussian_kde(log_returns, bw_method=bandwidth)
    samples = kde.resample(n_samples)[0]
    return samples

from sklearn.mixture import GaussianMixture

def sample_with_gmm(log_returns, n_samples, n_components=3):
    """
    Samples returns using a Gaussian Mixture Model.

    Args:
    - log_returns: Historical log returns.
    - n_samples: Number of samples to generate.
    - n_components: Number of Gaussian components.

    Returns:
    - samples: Samples drawn from the GMM.
    """
    log_returns = log_returns.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(log_returns)
    samples = gmm.sample(n_samples)[0].flatten()
    return samples

from multiprocessing import Pool


def simulate_jump_diffusion(args):
    """
    Simulates Jump Diffusion paths with Bayesian-estimated parameters for a single asset.

    Args:
    - args: Tuple containing (name, S0, trace, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump).

    Returns:
    - name: Asset name.
    - daily_means: Mean of simulated paths for each day.
    """
    name, S0, trace, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump = args
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = S0

    for t in range(1, n_steps):
        # Sample drift and volatility from posterior distributions
        mu = np.random.choice(trace['mu'])
        sigma = np.random.choice(trace['sigma'])

        # Brownian motion component
        z = np.random.normal(0, 1, size=n_paths)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z

        # Poisson process for jumps
        jumps = np.random.poisson(lambda_jump * dt, size=n_paths)
        jump_magnitudes = np.exp(np.random.normal(mu_jump, sigma_jump, size=n_paths)) - 1

        # Update paths
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion) * (1 + jumps * jump_magnitudes)

    # Calculate mean daily prices
    daily_means = np.mean(paths, axis=0).tolist()

    return name, daily_means


def jump_diffusion_multiple_assets(S0_dict, trace_dict, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump, n_processes=4):
    """
    Simulates Jump Diffusion paths for multiple assets and returns mean daily prices.

    Args:
    - S0_dict: Dictionary with asset names as keys and initial prices as values.
    - trace_dict: Dictionary with asset names as keys and MCMC traces as values.
    - T: Time horizon (years).
    - dt: Time step size.
    - n_paths: Number of paths.
    - lambda_jump: Jump intensity.
    - mu_jump: Mean jump size.
    - sigma_jump: Volatility of jump size.
    - n_processes: Number of parallel processes.

    Returns:
    - result_dict: Dictionary with asset names as keys and lists of mean daily prices as values.
    """
    # Prepare arguments for parallel processing
    args = [
        (name, S0_dict[name], trace_dict[name], T, dt, n_paths, lambda_jump, mu_jump, sigma_jump)
        for name in S0_dict
    ]

    # Parallelize simulation across assets
    with Pool(n_processes) as pool:
        results = pool.map(simulate_jump_diffusion, args)

    # Collect results into a dictionary
    result_dict = {name: daily_means for name, daily_means in results}

    return result_dict

import numpy as np
from multiprocessing import Pool
import os

def simulate_jump_diffusion_alt(args):
    """
    Simulates Jump Diffusion paths with Bayesian-estimated parameters for a single asset.

    Args:
    - args: Tuple containing (name, S0, trace, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump, seed).

    Returns:
    - name: Asset name.
    - daily_means: Mean of simulated paths for each day.
    """
    name, S0, trace, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump, seed = args

    # Seed the random number generator for this specific simulation
    np.random.seed(seed)

    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = S0

    for t in range(1, n_steps):
        # Sample drift and volatility from posterior distributions
        mu = np.random.choice(trace['mu'])
        sigma = np.random.choice(trace['sigma'])

        # Brownian motion component
        z = np.random.normal(0, 1, size=n_paths)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z

        # Poisson process for jumps
        jumps = np.random.poisson(lambda_jump * dt, size=n_paths)
        jump_magnitudes = np.exp(np.random.normal(mu_jump, sigma_jump, size=n_paths)) - 1

        # Update paths
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion) * (1 + jumps * jump_magnitudes)

    # Calculate mean daily prices
    daily_means = np.mean(paths, axis=0).tolist()

    return name, daily_means


def jump_diffusion_multiple_assets_alt(S0_dict, trace_dict, T, dt, n_paths, lambda_jump, mu_jump, sigma_jump, n_processes=4):
    """
    Simulates Jump Diffusion paths for multiple assets and returns mean daily prices.

    Args:
    - S0_dict: Dictionary with asset names as keys and initial prices as values.
    - trace_dict: Dictionary with asset names as keys and MCMC traces as values.
    - T: Time horizon (years).
    - dt: Time step size.
    - n_paths: Number of paths.
    - lambda_jump: Jump intensity.
    - mu_jump: Mean jump size.
    - sigma_jump: Volatility of jump size.
    - n_processes: Number of parallel processes.

    Returns:
    - result_dict: Dictionary with asset names as keys and lists of mean daily prices as values.
    """
    # Generate unique random seeds for each process
    random_seeds = [np.random.randint(0, 1e6) for _ in range(len(S0_dict))]

    # Prepare arguments for parallel processing
    args = [
        (name, S0_dict[name], trace_dict[name], T, dt, n_paths, lambda_jump, mu_jump, sigma_jump, seed)
        for name, seed in zip(S0_dict, random_seeds)
    ]

    # Parallelize simulation across assets
    with Pool(n_processes) as pool:
        results = pool.map(simulate_jump_diffusion_alt, args)

    # Collect results into a dictionary
    result_dict = {name: daily_means for name, daily_means in results}

    return result_dict



import matplotlib.pyplot as plt

def plot_real_vs_simulated(real_prices, simulated_prices, name):
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label="Real Prices", linewidth=2)
    plt.plot(simulated_prices, label="Simulated Prices", linestyle="--", linewidth=1.5)
    plt.title(f"Real vs Simulated Prices: {name}")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def run_bayesian_estimation_with_noise(shortData, n_processes=4, n_iterations=10, noise_scale=0.01):
    """
    Bayesian estimation with noise injection and averaging for multiple assets.

    Args:
    - shortData: Dictionary containing historical log returns for each asset.
    - n_processes: Number of parallel processes to use.
    - n_iterations: Number of noise-injection iterations.
    - noise_scale: Standard deviation of noise to inject into log returns.

    Returns:
    - averagedTraceDict: Dictionary with averaged posterior distributions for mu and sigma.
    """
    aggregated_traces = {name: {"mu": [], "sigma": []} for name in shortData}

    for _ in range(n_iterations):
        # Inject noise into log returns
        noisy_data = {
            name: {
                "log_returns": shortData[name]["log_returns"] + np.random.normal(
                    loc=0, scale=noise_scale, size=len(shortData[name]["log_returns"])
                )
            }
            for name in shortData
        }

        # Estimate Bayesian parameters
        traceDict = estimate_bayesian_parameters(noisy_data, n_processes=n_processes)

        # Aggregate the results
        for name, trace in traceDict.items():
            if trace:
                aggregated_traces[name]["mu"].extend(trace["mu"])
                aggregated_traces[name]["sigma"].extend(trace["sigma"])

    # Average results for each asset
    averagedTraceDict = {}
    for name, traces in aggregated_traces.items():
        averagedTraceDict[name] = {
            "mu": traces["mu"],
            "sigma": traces["sigma"],
        }

    return averagedTraceDict

def skewKurtVol(test_data, simulated_paths):
    """
    Compare skewness, kurtosis, and volatility between real test data and simulated paths.

    Args:
    - test_data: Dictionary containing the test data for each asset (real prices).
    - simulated_paths: Dictionary where each asset name maps to full simulated price paths (not just means).

    Returns:
    - None (prints comparisons).
    """

    # Flatten simulated paths for distribution metrics
    sim_prices_flat = np.ravel(simulated_paths)

    # Compare volatility
    simulated_volatility = np.std(sim_prices_flat)
    real_volatility = np.std(test_data)

    # Compare skewness and kurtosis
    simulated_skew = skew(sim_prices_flat)
    real_skew = skew(test_data)

    simulated_kurtosis = kurtosis(sim_prices_flat)
    real_kurtosis = kurtosis(test_data)

    print(f"  Volatility - Simulated: {simulated_volatility:.4f}, Real: {real_volatility:.4f}")
    print(f"  Skewness  - Simulated: {simulated_skew:.4f}, Real: {real_skew:.4f}")
    print(f"  Kurtosis  - Simulated: {simulated_kurtosis:.4f}, Real: {real_kurtosis:.4f}")


def send_it():
    #asset data
    filesCsv = [
        'apple.csv',
        'amazon.csv',
        'asml.csv'
    ]

    start_time = time.time()

    #corresponding dividend return in as decimal
    dividends = [0.0, 0.0, 0.0]
    # dividends must correspond to the number of assets
    if len(filesCsv) != len(dividends):
        raise ValueError("Number of files and dividends must be equal")

    """
    dictionary with names as keys for assets and files for csv file of data 
    """
    csvDict = {}
    csvDict['names'] = [file.replace('.csv', '') for file in filesCsv]
    csvDict['files'] = filesCsv
    numAssets = len(filesCsv)
    csvDict['numberOfAsset'] = numAssets

    n_simulations = 1000  # 1000000
    n_paths = 1000  # 100000
    outerSim = 5  # 1000
    # limit the optimized portfolio volatility to this
    maxVolatility = 0.15
    T = 1
    data_dict = csvRead(csvDict, dividends)
    dt = 1 / 252  # single trading day
    lambda_jump = 0.1  # 10% chance of a jump per year
    mu_jump = 0.02  # Mean jump size
    sigma_jump = 0.05  # Jump size volatility

    # Run the simulation
    mosey = [990.14, 82.64, 177.92]
    lilMosey = [988.20, 82.44, 177.81]
    S0 = []
    S0_dict = {}
    shortData = {}
    s0 = []
    start = []
    names = []
    for name in csvDict['names']:
        S0_dict[name] = data_dict[name]['prices'][-1]
        S0.append(data_dict[name]['prices'][-1])
        names.append(name)
        #shortList = data_dict[name]['prices'][:-252]
        shortList = data_dict[name]['prices'][:-504]
        start.append(shortList[-1])
        s0.append(shortList[-1])
        #shortLog = data_dict[name]['log_returns'][:-252]
        shortLog = data_dict[name]['log_returns'][:-504]
        shortData[name] = {
            'prices': shortList,
            'log_returns': shortLog
        }


    # Train-test split with 2 years for testing
    train_data = {}
    test_data = {}
    for name in shortData.keys():
        split_idx = len(shortData[name]['log_returns']) - 504  # Last 2 years
        train_data[name] = {
            'log_returns': shortData[name]['log_returns'][:split_idx],
            'prices': shortData[name]['prices'][:split_idx]
        }
        test_data[name] = {
            'log_returns': shortData[name]['log_returns'][split_idx:],
            'prices': shortData[name]['prices'][split_idx:]
        }

    # Loop through assets
    simulated_prices_dict = {}



    """
    train_data = {}
    test_data = {}
    for name in shortData.keys():
        # Use 80% of data for training and 20% for testing
        train_data[name] = {
            'log_returns': shortData[name]['log_returns'][:int(len(shortData[name]['log_returns']) * 0.8)],
            'prices': shortData[name]['prices'][:int(len(shortData[name]['prices']) * 0.8)]
        }
        test_data[name] = {
            'log_returns': shortData[name]['log_returns'][int(len(shortData[name]['log_returns']) * 0.8):],
            'prices': shortData[name]['prices'][int(len(shortData[name]['prices']) * 0.8):]
        }

    """
    est_bae = estimate_bayesian_parameters(train_data, n_processes=4)
    # Simulate prices using parameters from est_bae
    simulated_prices = jump_diffusion_multiple_assets(
    S0_dict={name: train_data[name]['prices'][-1] for name in train_data.keys()},
    trace_dict=est_bae,
    T=0.2,  # Assuming 20% of the data is roughly 0.2 years
    dt=1 / 252,  # Daily steps
    n_paths=1000,  # Number of paths to simulate
    lambda_jump=0.1,
    mu_jump=0.02,
    sigma_jump=0.05,
    n_processes=4
    )
    return est_bae

if __name__ == "__main__":
    #asset data
    filesCsv = [
        'apple.csv',
        'amazon.csv',
        'asml.csv'
    ]

    start_time = time.time()

    #corresponding dividend return in as decimal
    dividends = [0.0, 0.0, 0.0]
    # dividends must correspond to the number of assets
    if len(filesCsv) != len(dividends):
        raise ValueError("Number of files and dividends must be equal")

    """
    dictionary with names as keys for assets and files for csv file of data 
    """
    csvDict = {}
    csvDict['names'] = [file.replace('.csv', '') for file in filesCsv]
    csvDict['files'] = filesCsv
    numAssets = len(filesCsv)
    csvDict['numberOfAsset'] = numAssets

    n_simulations = 1000  # 1000000
    n_paths = 1000  # 100000
    outerSim = 5  # 1000
    # limit the optimized portfolio volatility to this
    maxVolatility = 0.15
    T = 1
    data_dict = csvRead(csvDict, dividends)
    dt = 1 / 252  # single trading day
    lambda_jump = 0.1  # 10% chance of a jump per year
    mu_jump = 0.02  # Mean jump size
    sigma_jump = 0.05  # Jump size volatility

    # Run the simulation
    mosey = [990.14, 82.64, 177.92]
    lilMosey = [988.20, 82.44, 177.81]
    S0 = []
    S0_dict = {}
    shortData = {}
    s0 = []
    start = []
    names = []
    for name in csvDict['names']:
        S0_dict[name] = data_dict[name]['prices'][-1]
        S0.append(data_dict[name]['prices'][-1])
        names.append(name)
        #shortList = data_dict[name]['prices'][:-252]
        shortList = data_dict[name]['prices'][:-504]
        start.append(shortList[-1])
        s0.append(shortList[-1])
        #shortLog = data_dict[name]['log_returns'][:-252]
        shortLog = data_dict[name]['log_returns'][:-504]
        shortData[name] = {
            'prices': shortList,
            'log_returns': shortLog
        }


    # Train-test split with 2 years for testing
    train_data = {}
    test_data = {}
    for name in shortData.keys():
        split_idx = len(shortData[name]['log_returns']) - 504  # Last 2 years
        train_data[name] = {
            'log_returns': shortData[name]['log_returns'][:split_idx],
            'prices': shortData[name]['prices'][:split_idx]
        }
        test_data[name] = {
            'log_returns': shortData[name]['log_returns'][split_idx:],
            'prices': shortData[name]['prices'][split_idx:]
        }


    # Loop through assets
    simulated_prices_dict = {}



    """
    train_data = {}
    test_data = {}
    for name in shortData.keys():
        # Use 80% of data for training and 20% for testing
        train_data[name] = {
            'log_returns': shortData[name]['log_returns'][:int(len(shortData[name]['log_returns']) * 0.8)],
            'prices': shortData[name]['prices'][:int(len(shortData[name]['prices']) * 0.8)]
        }
        test_data[name] = {
            'log_returns': shortData[name]['log_returns'][int(len(shortData[name]['log_returns']) * 0.8):],
            'prices': shortData[name]['prices'][int(len(shortData[name]['prices']) * 0.8):]
        }

    """
    window = 0.2  # % of the data for testing
    recent = 20
    train_data, test_data, data_dict, T, S0_dict =  dict(filesCsv, dividends, window, recent)
    est_bae = estimate_bayesian_parameters(train_data, n_processes=4)
    # Simulate prices using parameters from est_bae
    simulated_prices = jump_diffusion_multiple_assets_alt(
    S0_dict={name: train_data[name]['prices'][-1] for name in train_data.keys()},
    trace_dict=est_bae,
    T=0.2,  # Assuming 20% of the data is roughly 0.2 years
    dt=1 / 252,  # Daily steps
    n_paths=1000,  # Number of paths to simulate
    lambda_jump=0.1,
    mu_jump=0.02,
    sigma_jump=0.05,
    n_processes=4
    )


    """
    for name in test_data:
        real_prices = test_data[name]['prices']
        simulated_pric = simulated_prices[name][:len(real_prices)]
        plot_real_vs_simulated(real_prices, simulated_pric, name)
    
    for name in test_data:
        real_returns = test_data[name]['log_returns']
        simulated_price = simulated_prices[name]  # Simulated prices from jump diffusion
        simulated_returns = np.diff(np.log(simulated_price))

        print(f"Asset: {name}")
        print(f"Real Skewness: {skew(real_returns)}, Simulated Skewness: {skew(simulated_returns)}")
        print(f"Real Kurtosis: {kurtosis(real_returns)}, Simulated Kurtosis: {kurtosis(simulated_returns)}")
    """
    # Compare simulated prices to test_data prices
    for name in test_data.keys():
        real_price = test_data[name]['prices'][0]  # First price in the test set
        simulated_mean_price = np.mean(simulated_prices[name])
        print(f"{name}: Simulated Mean: {simulated_mean_price}, Real: {real_price}")

    """"
    for name in test_data.keys():
        # Compare volatility
        simulated_volatility = np.std(simulated_prices[name])
        real_volatility = np.std(test_data[name]['prices'])

        # Compare skewness and kurtosis
        simulated_skew = skew(simulated_prices[name])
        real_skew = skew(test_data[name]['prices'])

        simulated_kurtosis = kurtosis(simulated_prices[name])
        real_kurtosis = kurtosis(test_data[name]['prices'])

        print(f"{name}:")
        print(f"  Volatility - Simulated: {simulated_volatility}, Real: {real_volatility}")
        print(f"  Skewness  - Simulated: {simulated_skew}, Real: {real_skew}")
        print(f"  Kurtosis  - Simulated: {simulated_kurtosis}, Real: {real_kurtosis}")

    window_size = 504  # 2 years
    test_size = 126  # 6 months

    for start in range(0, len(shortData['apple']['log_returns']) - window_size - test_size, test_size):
        # Create training and testing splits
        train_data = {
            name: {
                'log_returns': shortData[name]['log_returns'][start:start + window_size],
                'prices': shortData[name]['prices'][start:start + window_size]
            }
            for name in shortData.keys()
        }
        test_data = {
            name: {
                'log_returns': shortData[name]['log_returns'][start + window_size:start + window_size + test_size],
                'prices': shortData[name]['prices'][start + window_size:start + window_size + test_size]
            }
            for name in shortData.keys()
        }

        # Estimate parameters and validate
        est_bae = estimate_bayesian_parameters(train_data, n_processes=4)
        simulated_prices = jump_diffusion_multiple_assets(
            S0_dict={name: train_data[name]['prices'][-1] for name in train_data.keys()},
            trace_dict=est_bae,
            T=test_size / 252,
            dt=1 / 252,
            n_paths=1000,
            lambda_jump=0.1,
            mu_jump=0.02,
            sigma_jump=0.05,
            n_processes=4
        )

    for name in test_data.keys():
        print(f"{name}: Mean Simulated Price: {np.mean(simulated_prices[name])}, Real: {test_data[name]['prices'][0]}")

    window_size = 504  # 2 years
    test_size = 126  # 6 months

    for start in range(0, len(shortData['apple']['log_returns']) - window_size - test_size, test_size):
        # Create training and testing splits
        train_data = {
            name: {
                'log_returns': shortData[name]['log_returns'][start:start + window_size],
                'prices': shortData[name]['prices'][start:start + window_size]
            }
            for name in shortData.keys()
        }
        test_data = {
            name: {
                'log_returns': shortData[name]['log_returns'][start + window_size:start + window_size + test_size],
                'prices': shortData[name]['prices'][start + window_size:start + window_size + test_size]
            }
            for name in shortData.keys()
        }

        # Estimate parameters and validate
        est_bae = estimate_bayesian_parameters(train_data, n_processes=4)
        simulated_prices = jump_diffusion_multiple_assets(
            S0_dict={name: train_data[name]['prices'][-1] for name in train_data.keys()},
            trace_dict=est_bae,
            T=test_size / 252,
            dt=1 / 252,
            n_paths=1000,
            lambda_jump=0.1,
            mu_jump=0.02,
            sigma_jump=0.05,
            n_processes=4
        )

        for name in test_data.keys():
            print(
                f"{name}: Mean Simulated Price: {np.mean(simulated_prices[name])}, Real: {test_data[name]['prices'][0]}")

    """
    elapsed_time = round(time.time() - start_time, 2)

    print(f"Elapsed time: {elapsed_time} seconds")



import time
from workSpace import dict
import torch
from fitter import Fitter
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm, t, lognorm

SEED = 42
torch.manual_seed(SEED)  # For PyTorch
np.random.seed(SEED)

def find_best_distributions(log_returns):
    """
    Find the best-fit distribution for log returns using Fitter.

    Args:
    - log_returns: Historical log returns.

    Returns:
    - best_distribution (str): Name of the best-fit distribution.
    - best_params (dict): Parameters for the best-fit distribution.
    """
    distribution = [
        'norm','lognorm', 't'
    ]
    f = Fitter(log_returns, distributions=distribution)
    f.fit()
    best_fit = f.get_best()
    best_distribution = list(best_fit.keys())[0]
    best_params = best_fit[best_distribution]
    return best_distribution, best_params


def simulate_markov_stock_prices_gpu(log_returns, dt, S0, best_n_components_bic, n_simulations, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)

    n_days = int(1 / dt)
    log_returns = np.array(log_returns)

    model = GaussianHMM(n_components=best_n_components_bic, covariance_type="diag", n_iter=1000, random_state=SEED)
    model.fit(log_returns.reshape(-1, 1))

    means = model.means_.flatten()
    variances = np.sqrt(model.covars_).flatten()
    transition_matrix = model.transmat_

    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    drift = mu - (sigma ** 2) / 2

    best_distribution, best_params = find_best_distributions(log_returns)
    print(f"Best distribution: {best_distribution}, Parameters: {best_params}")

    simulated_log_returns = torch.zeros((n_days, n_simulations), dtype=torch.float32, device=device)
    regimes = torch.randint(0, len(means), (n_simulations,), device=device)

    for time_step in range(n_days):  # Renamed loop variable
        log_returns_t = torch.zeros(n_simulations, device=device)
        for regime_idx in range(len(means)):
            sampled_regime_mask = (regimes == regime_idx)

            if sampled_regime_mask.any():
                num_samples = sampled_regime_mask.sum().item()
                if num_samples > 0:
                    if best_distribution == "norm":
                        samples = norm.rvs(
                            loc=best_params["loc"],
                            scale=best_params["scale"],
                            size=num_samples,
                            random_state=np.random.default_rng(SEED)  # Ensure reproducibility for norm
                        )
                    elif best_distribution == "t":
                        samples = t.rvs(
                            df=best_params["df"],
                            loc=best_params["loc"],
                            scale=best_params["scale"],
                            size=num_samples
                        )
                    elif best_distribution == "lognorm":
                        samples = lognorm.rvs(
                            s=best_params["s"],
                            loc=best_params["loc"],
                            scale=best_params["scale"],
                            size=num_samples
                        )
                    else:
                        raise ValueError(f"Unsupported distribution: {best_distribution}")
                    samples = torch.tensor(samples, dtype=torch.float32, device=device)
                    scaled_samples = samples * torch.sqrt(torch.tensor(variances[regime_idx], device=device)) \
                                     + means[regime_idx]
                    scaled_samples += drift
                    log_returns_t[sampled_regime_mask] = scaled_samples

        simulated_log_returns[time_step, :] = log_returns_t

        transition_probs = torch.tensor(transition_matrix, device=device)
        regimes = torch.multinomial(transition_probs[regimes], num_samples=1).squeeze()

    cumulative_log_returns = torch.cumsum(simulated_log_returns, dim=0)
    all_stock_prices = S0 * torch.exp(cumulative_log_returns)
    avg_stock_prices = torch.mean(all_stock_prices, dim=1)

    return avg_stock_prices.cpu().numpy(), all_stock_prices.cpu().numpy()


def select_optimal_n_components(log_returns, max_components=10, covariance_type="diag", random_state=42):
    bic_values = []
    aic_values = []
    n_values = range(1, max_components + 1)

    # Ensure log_returns data is clean and valid
    log_returns = np.array(log_returns)
    log_returns = log_returns[~np.isnan(log_returns)]  # Remove NaN values
    log_returns = log_returns[np.isfinite(log_returns)]  # Remove infinite values

    if log_returns.size < 10:  # Check if there's enough data
        raise ValueError("Insufficient data size for training the HMM model.")

    # Standardize the log_returns data
    scaler = StandardScaler()
    log_returns_scaled = scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()

    for n in n_values:
        try:
            model = GaussianHMM(n_components=n, covariance_type=covariance_type, n_iter=1000, random_state=random_state)
            model.fit(log_returns_scaled.reshape(-1, 1))

            log_likelihood = model.score(log_returns_scaled.reshape(-1, 1))  # Log-likelihood of the model
            n_parameters = n * (n - 1) + 2 * n + 2 * n * 1  # Approximation of parameters
            data_points = log_returns_scaled.shape[0]

            # Compute BIC and AIC
            bic = n_parameters * np.log(data_points) - 2 * log_likelihood
            aic = 2 * n_parameters - 2 * log_likelihood

            bic_values.append(bic)
            aic_values.append(aic)
        except ValueError as e:
            print(f"An error occurred for n_components={n}: {e}")
            bic_values.append(np.inf)
            aic_values.append(np.inf)

    # Find the best n_components using BIC/AIC
    best_n_components_bic = n_values[np.argmin(bic_values)]
    best_n_components_aic = n_values[np.argmin(aic_values)]

    return best_n_components_bic, best_n_components_aic, bic_values, aic_values

def plot_simulated_values(avg_stock_prices, test_data, name):
    plt.plot(avg_stock_prices, label="Predicted Prices")
    plt.plot(test_data, label="Actual Prices")
    plt.legend()
    plt.title(f"Price Prediction vs Actual Prices for {name}")  # Add the name dynamically as a title
    plt.show()



if __name__ == "__main__":
    filesCsv = ['amazon.csv', 'apple.csv', 'asml.csv', 'googl.csv', 'ko.csv']
    names = [file.replace('.csv', '') for file in filesCsv]
    n_simulations = 2000
    start_time = time.time()

    dividends = [0.0,0.0,0.0,0.0,0.0]
    if len(filesCsv) != len(dividends):
        raise ValueError("Number of files and dividends must be equal")

    train_data, test_data, data_dict, T, S0_dict = dict(filesCsv, dividends, 0.05, 20)
    dt = 1 / 252
    portfolio = {}
    for name in names:
        log_returns = train_data[name]['log_returns']
        S0 = train_data[name]['S0']
        best_n_components_bic, best_n_components_aic, bic_values, aic_values = select_optimal_n_components(log_returns)
        portfolio[name] = {
            'best bic': best_n_components_bic,
            'best aic': best_n_components_aic,
            'bic':bic_values,
            'aic': aic_values
        }
        avg_stock_prices, all_stock_prices = simulate_markov_stock_prices_gpu(log_returns, dt, S0, best_n_components_bic, n_simulations)
        real_prices = test_data[name]['prices']
        plot_simulated_values(avg_stock_prices, real_prices, name)

    elapsed_time = round(time.time() - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds")