from scipy.stats import norm, halfnorm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from scipy.stats import kstest, norm
from priceSimulator import skewKurtVol
from scipy.stats import skew, kurtosis, gaussian_kde
import numpy as np

def bayesian_estimation_for_asset(name, log_returns, n_samples):
    try:
        mu_prior = norm(loc=np.mean(log_returns), scale=np.std(log_returns) / 2)
        sigma_prior = halfnorm(scale=np.std(log_returns))

        def log_likelihood(mu, sigma, data):
            sigma = max(sigma, 1e-6)
            return np.sum(norm.logpdf(data, loc=mu, scale=sigma))

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
        sampled_sigma = sigma_samples[indices]  # Annualize sigma

        #print(f"{name}: Posterior Mu mean={np.mean(sampled_mu)}, Sigma mean={np.mean(sampled_sigma)}")
        trace = {"mu": sampled_mu, "sigma": sampled_sigma}
        return name, trace

    except Exception as e:
        print(f"Error processing {name}: {e}")
        return name, None




def estimate_bayesian_parameters(shortData, n_samples, n_processes=4):
    """
    Bayesian estimation of drift (mu) and volatility (sigma) for multiple assets using SciPy.

    Args:
    - shortData: Dictionary containing historical log returns for each asset.
    - n_processes: Number of parallel processes to use.

    Returns:
    - traceDict: Dictionary with asset names as keys and sampled posterior distributions for mu and sigma.
    """
    # Prepare arguments for parallel processing

    args = [(name, np.array(shortData[name]['log_returns']), n_samples) for name in shortData]

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

def gbmPrice_parallel(S0, mu, sigma, T, dt, n_paths, max_workers=4):
    for index, m in enumerate(mu):
        mu[index] = m * 252
    n_steps = int(T / dt)
    if len(mu) != len(sigma):
        raise ValueError(
            f"Length mismatch: mu (length={len(mu)}) and sigma (length={len(sigma)}) must have the same size.")

    args = [
        (mu[index], sigma[index], dt, n_steps)
        for index in range(len(mu))
        for _ in range(n_paths)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        final_prices = list(executor.map(simulate_gbm_log_diff_tf, args))

    return np.mean(final_prices)


def simulate_gbm_tf(args, device="/GPU:0"):
    """
    Simulates GBM paths using TensorFlow with GPU support.

    Args:
    - S0: Initial price (scalar).
    - mu: Drift values (annualized, list or array).
    - sigma: Volatility values (annualized, list or array).
    - T: Time horizon (in years).
    - dt: Time step size.
    - n_paths: Number of paths to simulate per set of parameters.
    - device: Device to run the simulation (e.g., "/GPU:0" or "/CPU:0").

    Returns:
    - Mean final prices of all simulated paths for each (mu, sigma) combination.
    """
    mu, sigma, dt, n_steps = args
    mu = tf.constant(mu, dtype=tf.float32) * 252  # Annualized drift
    sigma = tf.constant(sigma, dtype=tf.float32)  # Annualized volatility

    with tf.device(device):
        # Generate random numbers
        rand = tf.random.normal((len(mu), n_steps, n_steps), dtype=tf.float32)
        drift = (mu[:, None] - 0.5 * sigma[:, None]**2) * dt
        diffusion = sigma[:, None] * tf.sqrt(tf.constant(dt, dtype=tf.float32))

        # Calculate log returns
        log_returns = drift[:, :, None] + diffusion[:, :, None] * rand
        cumulative_returns = tf.cumsum(log_returns, axis=2)

        # Compute final prices
        final_prices = S0 * tf.exp(cumulative_returns[:, :, -1])
        mean_final_prices = tf.reduce_mean(final_prices, axis=1)

    return mean_final_prices.numpy()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, gaussian_kde, probplot

def analyze_log_returns(log_returns):
    # Convert log_returns to a NumPy array if it's not already
    log_returns = np.array(log_returns)

    # Compute descriptive statistics
    mean_return = np.mean(log_returns)
    std_return = np.std(log_returns)
    skewness = skew(log_returns)
    kurt = kurtosis(log_returns)
    """
    print("Descriptive Statistics:")
    print(f"Mean: {mean_return}")
    print(f"Standard Deviation: {std_return}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")
    """

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(log_returns, bins=50, density=True, alpha=0.6, color='g', label="Log Returns")

    # Overlay normal distribution
    x = np.linspace(np.min(log_returns), np.max(log_returns), 1000)
    plt.plot(x, norm.pdf(x, mean_return, std_return), 'k', linewidth=2, label="Normal Distribution")

    # Kernel Density Estimation
    kde = gaussian_kde(log_returns)
    plt.plot(x, kde(x), 'r--', label="KDE")

    plt.title("Log Returns Distribution")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()

    # Q-Q Plot
    plt.figure(figsize=(8, 6))
    probplot(log_returns, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Log Returns")
    plt.grid()
    plt.show()


import tensorflow as tf

def simulate_gbm_log_diff_tf(args):
    """
    Simulate Geometric Brownian Motion (GBM) paths and compute average daily log differences.

    Args:
        S0 (float): Initial stock price.
        mu (list): List of annualized drift rates for each asset.
        sigma (list): List of annualized volatilities for each asset.
        T (float): Time horizon in years.
        dt (float): Time step in years (1/252 for daily simulation).
        n_paths (int): Number of simulation paths.

    Returns:
        tf.Tensor: Average daily log differences for each asset.
    """
    mu, sigma, dt, n_steps = args

    # Generate random numbers
    rand = tf.random.normal((1, n_steps, n_steps), dtype=tf.float32)
    drift = (tf.expand_dims(mu, axis=1) - 0.5 * tf.expand_dims(sigma, axis=1)**2) * dt
    diffusion = tf.expand_dims(sigma, axis=1) * tf.sqrt(tf.constant(dt, dtype=tf.float32))

    # Calculate log differences (daily log returns)
    log_differences = drift[:, :, None] + diffusion[:, :, None] * rand

    # Compute average log differences across paths
    mean_daily_log_differences = tf.reduce_mean(log_differences, axis=1)  # Average over paths

    return mean_daily_log_differences




import numpy as np


def gbmPrice_parallel_single(S0, mu, sigma, T, dt, num_paths=1000):
    """
    Simulate Geometric Brownian Motion (GBM) paths and calculate mean prices at each time step
    for a single asset.

    Args:
        S0 (float): Initial stock price.
        mu (float): Annualized drift rate for the asset.
        sigma (float): Annualized volatility for the asset.
        T (float): Total simulation time in years.
        dt (float): Time step in years.
        num_paths (int): Number of Monte Carlo paths.

    Returns:
        np.ndarray: Mean prices across all simulated paths at each time step.
    """
    # Annualize drift
    mu = mu * 252  # Assume 252 trading days in the year
    n_steps = int(T / dt)  # Total number of time steps

    # Initialize array to hold asset prices for all paths and time steps
    values = np.zeros((n_steps, num_paths))

    # Set initial price for all simulated paths
    values[0, :] = S0

    # Simulate GBM paths
    for t in range(1, n_steps):
        # Compute drift and diffusion terms
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        # Generate random standard normal variables for all paths at step `t`
        z = np.random.normal(0, 1, num_paths)
        # Update asset prices at time step `t` for all paths
        values[t, :] = values[t - 1, :] * np.exp(drift + diffusion * z)

    # Compute mean prices across all paths at each time step
    mean_prices = np.mean(values, axis=1)  # Average over paths
    print(mean_prices[-1])

    return mean_prices


import numpy as np


def gbmLogReturns(S0, mu, sigma, T, dt, num_paths=1000):
    """
    Simulate Geometric Brownian Motion (GBM) paths and calculate mean log returns at each time step.

    Args:
        S0 (float): Initial stock price.
        mu (float): Annualized drift rate for the asset.
        sigma (float): Annualized volatility for the asset.
        T (float): Total simulation time in years.
        dt (float): Time step in years.
        num_paths (int): Number of Monte Carlo paths.

    Returns:
        np.ndarray: Mean log returns across all simulated paths at each time step.
    """
    # Ensure drift is in correct units
    mu = mu * 252  # Annualized drift to daily if 252 assumed trading days
    n_steps = int(T / dt)  # Total number of time steps
    sigma = sigma * np.sqrt(1 / dt)

    # Initialize array for all GBM paths
    values = np.zeros((n_steps, num_paths))
    values[0, :] = S0  # Set initial price for all paths

    # Simulate GBM paths step-by-step
    for t in range(1, n_steps):
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        z = np.random.normal(0, 1, num_paths)
        values[t, :] = values[t - 1, :] * np.exp(drift + diffusion * z)

    # Calculate log returns: log(S[t] / S[t-1])
    log_returns = np.log(values[1:, :] / values[:-1, :])

    # Compute mean log return across all paths at each time step
    mean_log_returns = np.mean(log_returns, axis=1)
    #print(mean_log_returns[-1])

    return mean_log_returns


from scipy.stats import skew, kurtosis
from arch import arch_model

def hrs_single(log_returns, T, dt, S0):
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


    # Retrieve historical log returns
    log_returns = np.array(log_returns, dtype=np.float32)

    # Rescale log returns
    scale_factor = 100
    scaled_returns = log_returns * scale_factor


    skewness = skew(log_returns)
    kurt = kurtosis(log_returns)


    # Fit GARCH(1,1) model to the rescaled log returns
    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp="off")
    sz = 500
    all_simulated_prices = np.zeros((sz, n_steps))
    simulated_data = []
    analyze_log_returns(log_returns)
    # Simulate GARCH returns (ensure these are simple returns)
    simulated_data = garch_model.simulate(params=garch_fit.params, nobs=n_steps)
    # Convert simple returns to log-returns if needed
    simulated_returns = np.log(1 + simulated_data['data'] / scale_factor)
    #skewKurtVol(log_returns, simulated_returns)
    analyze_log_returns(simulated_returns)
    skewKurtVol(log_returns, simulated_returns)
    stat, p = kstest(simulated_returns, 'norm', args=(np.mean(simulated_returns), np.std(simulated_returns)))
    print(f"Statistic: {stat}, p-value: {p}")
    if p > 0.05:
        print("Fail to reject H0: Data appears normal.")
    else:
        print("Reject H0: Data is not normal.")
    for i in range(sz):
        for step in range(n_steps):
            # Calculate cumulative returns
            cumulative_returns = np.cumsum(simulated_returns)

            # Simulate daily prices using the cumulative log-returns
            daily_prices = S0 * np.exp(cumulative_returns)

            # Store all simulated daily prices
            all_simulated_prices[i, :] = daily_prices
    mean_daily_prices = np.mean(all_simulated_prices, axis=0).tolist()

    print('predicted price',mean_daily_prices[-1])


    return mean_daily_prices


def hrs_single_parallel(log_returns, T, dt, S0):
    """
    Simulates historical returns using a GARCH(1,1) model for assets and computes penalty metrics.
    This version is parallelized using TensorFlow and supports GPU acceleration (via tensorflow-metal).

    Args:
    - log_returns (list or array): Historical log returns for the asset.
    - T (float): Total simulation time in years.
    - dt (float): Time step size as a fraction of a year.
    - S0 (float): Initial price of the asset.

    Returns:
    - list: Mean daily prices across all simulations.
    """

    # Simulation parameters
    n_steps = int(T / dt)
    n_simulations = 10000  # Number of simulations (batch size)
    S0 = float(S0)  # Ensure S0 is scalar

    # Convert log returns to NumPy array for processing
    log_returns = np.array(log_returns, dtype=np.float32)

    # Rescale log returns for GARCH model fitting
    scale_factor = 100
    scaled_returns = log_returns * scale_factor

    # Fit GARCH(1,1) model to the rescaled log returns
    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp="off")

    # Generate unique simulated data and returns for each simulation
    all_simulated_returns = []
    for _ in range(n_simulations):
        simulated_data = garch_model.simulate(params=garch_fit.params, nobs=n_steps)
        simulated_returns = np.log(1 + simulated_data['data'] / scale_factor)
        all_simulated_returns.append(simulated_returns)

    # Stack all simulated returns into a 2D array (shape: [n_simulations, n_steps])
    all_simulated_returns = np.vstack(all_simulated_returns).astype(np.float32)

    # Convert required data to TensorFlow tensors for GPU compute
    simulated_returns_tf = tf.constant(all_simulated_returns, dtype=tf.float32)  # Shape: (n_simulations, n_steps)
    S0_tf = tf.constant(S0, dtype=tf.float32)

    # Compute cumulative prices for all simulations
    cumulative_returns = tf.cumsum(simulated_returns_tf, axis=1)
    cumulative_prices = S0_tf * tf.exp(cumulative_returns)

    # Calculate mean daily prices across all simulations
    mean_daily_prices = tf.reduce_mean(cumulative_prices, axis=0)

    # Convert result to NumPy for output
    mean_daily_prices_np = mean_daily_prices.numpy()
    print(f"Predicted price (final step): {mean_daily_prices_np[-1]}")
    return mean_daily_prices_np.tolist()

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
    mu_sum = 0
    lambda_sum = 0
    sigma_sum = 0
    initial_lambda_jump = lambda_jump
    initial_mu_jump = mu_jump
    initial_sigma_jump = sigma_jump
    for t in range(1, n_steps):
        """
        if t % 252 == 0:  # Reset annually
            lambda_jump = initial_lambda_jump
            mu_jump = initial_mu_jump
            sigma_jump = initial_sigma_jump

        # Apply perturbations quarterly
        if t % (252 // 4) == 0:
            lambda_jump *= np.random.uniform(0.85, 1.0)
            mu_jump *= np.random.uniform(0.85, 1.0)
            sigma_jump *= np.random.uniform(0.98, 1.0)

        # Enforce lower bounds
        lambda_jump = max(lambda_jump, initial_lambda_jump * 0.5)
        mu_jump = max(mu_jump, initial_mu_jump * 0.5)
        sigma_jump = max(sigma_jump, initial_sigma_jump * 0.5)
        """
        mu_sum += mu_jump
        lambda_sum += lambda_jump
        sigma_sum += sigma_jump
        # Sample drift and volatility from posterior distributions
        mu = np.random.choice(trace['mu']) * 252
        sigma = np.random.choice(trace['sigma']) * sqrt(1 / 252)

        # Brownian motion component
        z = np.random.normal(0, 1, size=n_paths)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z

        # Poisson process for jumps
        jumps = np.random.poisson(lambda_jump * dt, size=n_paths)
        jump_magnitudes = np.exp(np.random.normal(mu_jump, sigma_jump, size=n_paths)) - 1
        # Update paths
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion) * (1 + jumps * jump_magnitudes)
        #if t % 252 == 0:
            #print(paths[:, t - 1] * np.exp(drift + diffusion) * (1 + jumps * jump_magnitudes))
    # Calculate mean daily prices
    daily_means = np.mean(paths, axis=0).tolist()
    #print(name,paths[0][-5:], paths[0][:5])
    mu_sum /= (n_steps - 1)
    lambda_sum /= (n_steps - 1)
    sigma_sum /= (n_steps - 1)
    #print('mu', mu_sum, 'lambda', lambda_sum, 'sigma', sigma_sum)

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
    #print('lambda_jump',lambda_jump, 'mu_jump', mu_jump, 'sigma_jump', sigma_jump)
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


import numpy as np
from scipy.stats import norm


def calculate_jump_parameters(log_returns, threshold=3):
    """
    Calculate jump diffusion parameters (lambda_jump, mu_jump, sigma_jump) from historical log returns.

    Args:
    - log_returns: Array of historical log returns.
    - threshold: Number of standard deviations to classify a jump.

    Returns:
    - lambda_jump: Jump intensity (jumps per year).
    - mu_jump: Mean jump size.
    - sigma_jump: Volatility of jump sizes.
    """
    # Fit normal distribution to returns
    log_returns = np.array(log_returns)
    mu, sigma = np.mean(log_returns), np.std(log_returns)

    # Identify jumps
    jump_indices = np.where(np.abs(log_returns - mu) > threshold * sigma)[0]
    jump_returns = log_returns[jump_indices]

    # Calculate lambda_jump
    T = len(log_returns) / 252  # Assuming 252 trading days per year
    lambda_jump = len(jump_returns)

    # Calculate mu_jump and sigma_jump
    if len(jump_returns) > 0:
        mu_jump = np.mean(jump_returns)
        sigma_jump = np.std(jump_returns)
    else:
        mu_jump = 0
        sigma_jump = 0
        print("No jumps detected")

    return lambda_jump, mu_jump, sigma_jump

def calculate_jump_parameters_alt(log_returns, threshold=3.4, method="std"):
    """
    Calculate jump diffusion parameters (lambda_jump, mu_jump, sigma_jump) from historical log returns.

    Args:
    - log_returns: Array of historical log returns.
    - threshold: Number of standard deviations or percentile range to classify a jump.
    - method: "std" for standard deviation threshold or "percentile" for percentile-based threshold.

    Returns:
    - lambda_jump: Jump intensity (jumps per year).
    - mu_jump: Mean jump size.
    - sigma_jump: Volatility of jump sizes.
    """
    log_returns = np.array(log_returns)
    mu, sigma = np.mean(log_returns), np.std(log_returns)

    if method == "std":
        # Identify jumps based on standard deviation threshold
        jump_indices = np.where(np.abs(log_returns - mu) > threshold * sigma)[0]
    elif method == "percentile":
        # Identify jumps based on percentiles
        lower_threshold = np.percentile(log_returns, 2.5)  # Bottom 2.5%
        upper_threshold = np.percentile(log_returns, 97.5)  # Top 2.5%
        jump_indices = np.where((log_returns < lower_threshold) | (log_returns > upper_threshold))[0]
    else:
        raise ValueError("Method must be 'std' or 'percentile'.")

    jump_returns = log_returns[jump_indices]

    # Calculate lambda_jump
    T = len(log_returns) / 252  # Assuming 252 trading days per year
    lambda_jump = len(jump_returns) / T

    # Calculate mu_jump and sigma_jump
    if len(jump_returns) > 0:
        mu_jump = np.mean(jump_returns)
        sigma_jump = np.std(jump_returns)
    else:
        mu_jump = 0
        sigma_jump = 0
        print("No jumps detected")

    print(f"Calibrated Parameters: lambda_jump={lambda_jump}, mu_jump={mu_jump}, sigma_jump={sigma_jump}")
    return lambda_jump, mu_jump, sigma_jump

def calibrate_jump_parameters_to_converge(
    S0_dict,
    trace_dict,
    T,
    dt,
    n_paths,
    real_price,
    lambda_jump,
    mu_jump,
    sigma_jump,
    n_processes=4,
    max_iterations=100,
    tolerance=1.0
):
    """
    Calibrates jump diffusion parameters (lambda_jump, mu_jump, sigma_jump) by running simulations
    until the simulated price converges to the real price.

    Args:
    - S0_dict: Dictionary with asset names and starting prices.
    - trace_dict: Dictionary with asset names and Bayesian traces for mu and sigma.
    - T: Time horizon in years.
    - dt: Time step size.
    - n_paths: Number of paths.
    - real_price: The target real price for convergence.
    - lambda_jump: Initial jump intensity (jumps per year).
    - mu_jump: Initial mean jump size.
    - sigma_jump: Initial jump size volatility.
    - n_processes: Number of parallel processes.
    - max_iterations: Maximum number of iterations for the calibration loop.
    - tolerance: Tolerance for convergence (absolute difference between simulated and real price).

    Returns:
    - calibrated_lambda_jump, calibrated_mu_jump, calibrated_sigma_jump: Optimized parameters.
    """
    iteration = 0
    current_price = None
    adjustment_factor = 0.05  # Factor to adjust parameters per iteration

    while iteration < max_iterations:
        # Run simulation
        simulated_prices = jump_diffusion_multiple_assets_alt(
            S0_dict=S0_dict,
            trace_dict=trace_dict,
            T=T,
            dt=dt,
            n_paths=n_paths,
            lambda_jump=lambda_jump,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
            n_processes=n_processes
        )

        # Extract simulated price for Apple
        current_price = simulated_prices['apple'][-1]

        # Print progress
        print(
            f"Iteration {iteration + 1}: Simulated Price={current_price:.2f}, "
            f"Real Price={real_price:.2f}, Lambda={lambda_jump:.2f}, "
            f"Mu Jump={mu_jump:.5f}, Sigma Jump={sigma_jump:.5f}"
        )

        # Check convergence
        if abs(current_price - real_price) <= tolerance:
            print("Converged successfully!")
            break

        # Adjust parameters based on the direction of the error
        if current_price < real_price:
            lambda_jump *= 1 + adjustment_factor
            mu_jump *= 1 + adjustment_factor
            sigma_jump *= 1 + adjustment_factor
        else:
            lambda_jump *= 1 - adjustment_factor
            mu_jump *= 1 - adjustment_factor
            sigma_jump *= 1 - adjustment_factor

        iteration += 1

    if iteration == max_iterations:
        print("Max iterations reached. Convergence may not have been achieved.")

    return lambda_jump, mu_jump, sigma_jump

def jump_diffusion_with_calibration(
    S0_dict, trace_dict, T, dt, n_paths, real_prices,
    lambda_jump, mu_jump, sigma_jump, max_iterations=50, tolerance=1e-2, n_processes=4
):
    """
    Simulates Jump Diffusion paths and calibrates jump parameters until simulated prices converge to real prices.

    Args:
    - S0_dict: Dictionary with asset names as keys and initial prices as values.
    - trace_dict: Dictionary with asset names as keys and MCMC traces as values.
    - T: Time horizon (years).
    - dt: Time step size.
    - n_paths: Number of paths.
    - real_prices: Dictionary of real prices for each asset to calibrate against.
    - lambda_jump: Initial jump intensity.
    - mu_jump: Initial mean jump size.
    - sigma_jump: Initial jump size volatility.
    - max_iterations: Maximum number of calibration iterations.
    - tolerance: Tolerance for price convergence.
    - n_processes: Number of parallel processes.

    Returns:
    - result_dict: Dictionary with asset names as keys and calibrated jump parameters as values.
    """

    def single_simulation(args):
        name, S0, trace, lambda_jump, mu_jump, sigma_jump = args

        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0

        for t in range(1, n_steps):
            mu = np.random.choice(trace['mu']) * 252
            sigma = np.random.choice(trace['sigma']) * np.sqrt(1 / 252)

            z = np.random.normal(0, 1, size=n_paths)
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * z

            jumps = np.random.poisson(lambda_jump * dt, size=n_paths)
            jump_magnitudes = np.exp(np.random.normal(mu_jump, sigma_jump, size=n_paths)) - 1

            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion) * (1 + jumps * jump_magnitudes)

        # Return the mean of the final prices
        return name, np.mean(paths[:, -1])

    calibrated_params = {}

    for _ in range(max_iterations):
        # Prepare arguments for parallel processing
        args = [
            (name, S0_dict[name], trace_dict[name], lambda_jump, mu_jump, sigma_jump)
            for name in S0_dict.keys()
        ]

        # Parallelize the simulation
        with Pool(n_processes) as pool:
            simulated_prices = dict(pool.map(single_simulation, args))

        # Calculate the errors and update parameters
        converged = True
        for name in real_prices.keys():
            simulated_price = simulated_prices[name]
            real_price = real_prices[name]

            error = simulated_price - real_price
            if abs(error) > tolerance:
                converged = False
                lambda_jump *= 1.0 + 0.1 * np.sign(error)
                mu_jump *= 1.0 + 0.05 * np.sign(error)
                sigma_jump *= 1.0 + 0.05 * np.sign(error)

        if converged:
            break

    # Store calibrated parameters
    calibrated_params = {
        "lambda_jump": lambda_jump,
        "mu_jump": mu_jump,
        "sigma_jump": sigma_jump,
    }
    return simulated_prices, calibrated_params


import tensorflow as tf
import numpy as np


def simulate_garch_paths_tf_refined(log_returns, T, dt, S0, n_simulations=10000, scale_factor=100):
    """
    Simulates GARCH(1,1) paths on the GPU using TensorFlow with refined accuracy.

    Args:
    - log_returns: Historical log returns for the asset.
    - T: Total simulation time in years.
    - dt: Time step size as a fraction of a year.
    - S0: Initial price of the asset.
    - n_simulations: Number of simulation paths.
    - scale_factor: Factor to scale returns for GARCH fitting.

    Returns:
    - np.array: Mean daily prices across simulations.
    """
    # Simulation parameters
    n_steps = int(T / dt)
    log_returns = np.array(log_returns, dtype=np.float32)
    scaled_returns = log_returns * scale_factor

    # Fit GARCH(1,1) model (CPU fitting for accuracy)
    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp="off")

    # Extract parameters from the fitted model
    omega, alpha, beta = (
        garch_fit.params['omega'],
        garch_fit.params['alpha[1]'],
        garch_fit.params['beta[1]'],
    )
    initial_vol = np.var(scaled_returns)

    # Convert parameters to TensorFlow tensors
    omega = tf.constant(omega, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    beta = tf.constant(beta, dtype=tf.float32)
    initial_vol = tf.constant(initial_vol, dtype=tf.float32)

    # Initialize simulation tensors
    returns = tf.TensorArray(dtype=tf.float32, size=n_steps, dynamic_size=False)
    volatilities = tf.ones([n_simulations], dtype=tf.float32) * initial_vol
    innovations = tf.random.normal([n_simulations, n_steps], dtype=tf.float32)

    # Simulate GARCH paths iteratively
    for t in tf.range(n_steps):
        if t == 0:
            # Initialize the first step
            current_returns = tf.zeros([n_simulations], dtype=tf.float32)
        else:
            # Update volatilities and returns based on GARCH equations
            volatilities = tf.sqrt(omega + alpha * tf.square(current_returns) + beta * tf.square(volatilities))
            current_returns = volatilities * innovations[:, t]

        # Write current returns to TensorArray
        returns = returns.write(t, current_returns)

    # Stack all returns and compute cumulative returns
    all_returns = tf.transpose(returns.stack()) / scale_factor  # Shape: [n_simulations, n_steps]
    cumulative_returns = tf.cumsum(all_returns, axis=1)
    cumulative_prices = S0 * tf.exp(cumulative_returns)

    # Compute mean daily prices
    mean_daily_prices = tf.reduce_mean(cumulative_prices, axis=0)

    return mean_daily_prices.numpy()


import tensorflow as tf
import numpy as np


def simulate_garch_paths_tf(log_returns, T, dt, S0, n_simulations, scale_factor):
    """
    Simulates GARCH(1,1) paths on the GPU using TensorFlow with refined accuracy.



use pytorch!!!!!!!!!!!!






    Args:
    - log_returns: Historical log returns for the asset.
    - T: Total simulation time in years.
    - dt: Time step size as a fraction of a year.
    - S0: Initial price of the asset.
    - n_simulations: Number of simulation paths.
    - scale_factor: Factor to scale returns for GARCH fitting.

    Returns:
    - np.array: Mean daily prices across simulations.
    """
    # Simulation parameters
    n_steps = int(T / dt)
    log_returns = np.array(log_returns, dtype=np.float32)
    scaled_returns = log_returns * scale_factor

    # Fit GARCH(1,1) model (CPU fitting for accuracy)
    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp="off")

    # Extract parameters from the fitted model
    omega, alpha, beta = (
        garch_fit.params['omega'],
        garch_fit.params['alpha[1]'],
        garch_fit.params['beta[1]'],
    )
    initial_vol = np.var(scaled_returns)

    # Convert parameters to TensorFlow tensors
    omega = tf.constant(omega, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    beta = tf.constant(beta, dtype=tf.float32)
    initial_vol = tf.constant(initial_vol, dtype=tf.float32)

    # Initialize simulation tensors
    returns = tf.TensorArray(dtype=tf.float32, size=n_steps, dynamic_size=False)
    volatilities = tf.ones([n_simulations], dtype=tf.float32) * initial_vol
    innovations = tf.random.normal([n_simulations, n_steps], dtype=tf.float32)

    # Simulate GARCH paths iteratively
    for t in tf.range(n_steps):
        if t == 0:
            # Initialize the first step
            current_returns = tf.zeros([n_simulations], dtype=tf.float32)
        else:
            # Update volatilities and returns based on GARCH equations
            volatilities = tf.sqrt(omega + alpha * tf.square(current_returns) + beta * tf.square(volatilities))
            current_returns = volatilities * innovations[:, t]

        # Write current returns to TensorArray
        returns = returns.write(t, current_returns)

    # Stack all returns and compute cumulative returns
    all_returns = tf.transpose(returns.stack()) / scale_factor  # Shape: [n_simulations, n_steps]
    cumulative_returns = tf.cumsum(all_returns, axis=1)
    cumulative_prices = S0 * tf.exp(cumulative_returns)

    # Compute mean daily prices
    mean_daily_prices = tf.reduce_mean(cumulative_prices, axis=0)

    return mean_daily_prices.numpy()

import tensorflow as tf
import numpy as np
from arch import arch_model
import time


def simulate_garch_paths_tf_mini(log_returns, T, dt, S0, n_simulations, scale_factor, seed=None):
    """
    Simulates GARCH(1,1) paths on the GPU using TensorFlow, optimized for accuracy and GPU efficiency.

    Args:
    - log_returns: Historical log returns for the asset.
    - T: Total simulation time in years.
    - dt: Time step size as a fraction of a year.
    - S0: Initial price of the asset.
    - n_simulations: Number of simulation paths.
    - scale_factor: Factor to scale returns for GARCH fitting.
    - seed: Seed for reproducibility (optional).

    Returns:
    - np.array: Mean daily prices across simulations.
    """
    if seed is not None:  # Set TensorFlow seed for reproducibility
        tf.random.set_seed(seed)

    # Simulation parameters
    n_steps = int(T / dt)
    log_returns = np.array(log_returns, dtype=np.float32)
    scaled_returns = log_returns * scale_factor

    # Fit GARCH(1,1) model (CPU fitting for accuracy)
    garch_model = arch_model(scaled_returns, vol="Garch", p=1, q=1, dist="normal")
    garch_fit = garch_model.fit(disp="off")

    # Extract parameters from the fitted model
    omega, alpha, beta = (
        garch_fit.params["omega"],
        garch_fit.params["alpha[1]"],
        garch_fit.params["beta[1]"],
    )
    initial_vol = np.var(scaled_returns)

    # Convert parameters to TensorFlow tensors
    omega = tf.constant(omega, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    beta = tf.constant(beta, dtype=tf.float32)
    initial_vol = tf.constant(initial_vol, dtype=tf.float32)

    # Initialize tensors for simulation
    volatilities = tf.ones([n_simulations], dtype=tf.float32) * tf.sqrt(initial_vol)
    innovations = tf.random.normal((n_simulations, n_steps), dtype=tf.float32)
    current_returns = tf.zeros([n_simulations], dtype=tf.float32)
    returns = tf.TensorArray(dtype=tf.float32, size=n_steps)

    # Define GARCH simulation loop as TensorFlow computational graph
    @tf.function
    def simulate_garch_stepss():
        # Create TensorArray within the graph
        # Initialize tensors for volatility and current returns
        vol = volatilities
        curr_returns = current_returns
        returns = tf.TensorArray( size=n_steps, clear_after_read=False)

        # Simulation loop
        for t in tf.range(n_steps):
            vol = tf.sqrt(omega + alpha * tf.square(curr_returns) + beta * tf.square(vol))
            curr_returns = vol * innovations[:, t]

            # Write returns to the TensorArray
            returns = returns.write(t, curr_returns)

        # Convert TensorArray to a stacked tensor
        return tf.transpose(returns.stack()) / scale_factor  # Shape: [n_simulations, n_steps]

    # Run the simulation steps
    all_returns = simulate_garch_stepss()

    # Calculate cumulative returns and prices
    cumulative_returns = tf.cumsum(all_returns, axis=1)
    cumulative_prices = S0 * tf.exp(cumulative_returns)

    # Mean daily prices across all simulations
    mean_daily_prices = tf.reduce_mean(cumulative_prices, axis=0)

    # Return as NumPy array
    return mean_daily_prices.numpy()

def simulate_garch_paths_mps(log_returns, T, dt, S0, n_simulations=10000, scale_factor=100, device=None):
    """
    Simulates GARCH(1,1) paths on the GPU using PyTorch with MPS (Metal Performance Shaders).

    Args:
    - log_returns: Historical log returns for the asset.
    - T: Total simulation time in years.
    - dt: Time step size as a fraction of a year.
    - S0: Initial price of the asset.
    - n_simulations: Number of simulation paths.
    - scale_factor: Factor to scale returns for GARCH fitting.
    - device: Device to run the simulation on ('mps' for macOS GPUs, 'cpu' as fallback).

    Returns:
    - np.array: Mean daily prices across simulations.
    """
    # Determine device
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Simulation parameters
    n_steps = int(T / dt)
    log_returns = np.array(log_returns, dtype=np.float32)
    scaled_returns = log_returns * scale_factor

    # Fit GARCH(1,1) model (CPU fitting for accuracy)
    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp="off")

    # Extract parameters from the fitted model
    omega, alpha, beta = (
        garch_fit.params['omega'],
        garch_fit.params['alpha[1]'],
        garch_fit.params['beta[1]'],
    )
    initial_vol = np.var(scaled_returns)

    # Convert parameters to PyTorch tensors
    omega = torch.tensor(omega, dtype=torch.float32, device=device)
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
    beta = torch.tensor(beta, dtype=torch.float32, device=device)
    initial_vol = torch.tensor(initial_vol, dtype=torch.float32, device=device)

    # Initialize simulation tensors
    volatilities = torch.ones((n_simulations,), dtype=torch.float32, device=device) * initial_vol.sqrt()
    innovations = torch.randn((n_simulations, n_steps), dtype=torch.float32, device=device)
    returns = torch.zeros((n_simulations, n_steps), dtype=torch.float32, device=device)

    # Simulate GARCH paths iteratively
    for t in range(n_steps):
        if t == 0:
            current_returns = torch.zeros((n_simulations,), dtype=torch.float32, device=device)
        else:
            # Update volatilities and returns based on GARCH equations
            volatilities = torch.sqrt(omega + alpha * current_returns ** 2 + beta * volatilities ** 2)
            current_returns = volatilities * innovations[:, t]

        # Store current returns
        returns[:, t] = current_returns

    # Scale and compute cumulative returns
    scaled_returns = returns / scale_factor
    cumulative_returns = torch.cumsum(scaled_returns, dim=1)
    cumulative_prices = S0 * torch.exp(cumulative_returns)

    # Compute mean daily prices
    mean_daily_prices = torch.mean(cumulative_prices, dim=0)

    return mean_daily_prices.cpu().numpy()


def simulate_single_path(garch_model, params, n_steps, scale_factor):
    """
    Simulate a single path using the GARCH model.

    Args:
    - garch_model: Fitted GARCH model object.
    - params: Parameters for the GARCH model simulation.
    - n_steps: Number of time steps in the simulation.
    - scale_factor: Factor used to rescale returns.

    Returns:
    - np.array: Simulated log returns for a single path.
    """
    single_sim = garch_model.simulate(params=params, nobs=n_steps)
    return np.log(1 + single_sim["data"] / scale_factor)


def hrs_single_parallel_alt(log_returns, n_simulations, T, dt, S0):
    """
    Simulates historical returns using a GARCH(1,1) model for assets and computes penalty metrics.
    This version is parallelized using joblib and supports GPU acceleration (via tensorflow-metal).

    Args:
    - log_returns (list or array): Historical log returns for the asset.
    - T (float): Total simulation time in years.
    - dt (float): Time step size as a fraction of a year.
    - S0 (float): Initial price of the asset.

    Returns:
    - list: Mean daily prices across all simulations.
    """
    np.random.seed(42)  # Ensures consistent random numbers
    tf.random.set_seed(42)
    # Simulation parameters
    n_steps = int(T / dt)
    S0 = float(S0)  # Ensure S0 is scalar

    # Convert log returns to NumPy array for processing
    log_returns = np.array(log_returns, dtype=np.float32)

    # Rescale log returns for GARCH model fitting
    scale_factor = 100
    scaled_returns = log_returns * scale_factor

    # Fit GARCH(1,1) model to the rescaled log returns
    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp="off")

    # Parallelized simulation of all paths
    simulated_data = Parallel(n_jobs=-1)(
        delayed(simulate_single_path)(garch_model, garch_fit.params, n_steps, scale_factor)
        for _ in range(n_simulations)
    )

    # Convert simulated data to a NumPy array
    all_simulated_returns = np.array(simulated_data)  # Shape: (n_simulations, n_steps)

    # Convert required data to TensorFlow tensors for GPU compute
    simulated_returns_tf = tf.constant(all_simulated_returns, dtype=tf.float32)  # Shape: (n_simulations, n_steps)
    S0_tf = tf.constant(S0, dtype=tf.float32)

    # Compute cumulative prices for all simulations
    cumulative_returns = tf.cumsum(simulated_returns_tf, axis=1)
    cumulative_prices = S0_tf * tf.exp(cumulative_returns)

    # Calculate mean daily prices across all simulations
    mean_daily_prices = tf.reduce_mean(cumulative_prices, axis=0)

    # Convert result to NumPy for output
    mean_daily_prices_np = mean_daily_prices.numpy()
    print(f"Predicted price (final step): {mean_daily_prices_np[-1]}")
    return mean_daily_prices_np.tolist()


def simulate_markov_stock_prices(log_returns, dt, S0, n_simulations):
    """
    Simulates stock prices using a Markov regime-switching model with drift.

    Args:
        log_returns: Array of historical log returns.
        dt: Time step size (e.g., 1/252 for daily returns).
        S0: Initial stock price.
        n_simulations: Number of simulation paths.

    Returns:
        avg_stock_prices: Average stock prices across simulations (n_days,).
        all_stock_prices: Simulated stock prices (n_days x n_simulations).
    """
    # Calculate the number of days to simulate
    n_days = int(1 / dt)

    # Convert log returns to NumPy array
    log_returns = np.array(log_returns)

    # Fit a Hidden Markov Model (HMM) to the log returns
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(log_returns.reshape(-1, 1))

    # Extract regime parameters
    means = model.means_.flatten()  # Mean log returns for each regime
    variances = np.sqrt(model.covars_).flatten()  # Standard deviations for each regime
    transition_matrix = model.transmat_  # Regime transition probabilities

    # Compute drift term
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    drift = mu - (sigma ** 2) / 2

    # Initialize simulated log returns (n_days x n_simulations)
    simulated_log_returns = np.zeros((n_days, n_simulations))

    # Simulate log returns for all paths
    for j in range(n_simulations):
        regime = np.random.choice([0, 1])  # Random initial regime

        for t in range(n_days):
            # Generate log return based on the current regime and add drift
            log_return = np.random.normal(loc=means[regime], scale=variances[regime]) + drift
            simulated_log_returns[t, j] = log_return

            # Transition to the next regime
            regime = np.random.choice([0, 1], p=transition_matrix[regime])

    # Compute cumulative log returns along the row axis (days)
    cumulative_log_returns = np.cumsum(simulated_log_returns, axis=0)

    # Convert cumulative log returns to stock prices
    all_stock_prices = S0 * np.exp(cumulative_log_returns)

    # Average prices over all simulation paths (columns)
    avg_stock_prices = np.mean(all_stock_prices, axis=1)

    return avg_stock_prices, all_stock_prices


from hmmlearn.hmm import GMMHMM


def simulate_markov_stock_prices_gpu(log_returns, dt, S0, n_simulations, device=None):
    """
    Simulates stock prices using a Markov regime-switching model with GPU acceleration (PyTorch).

    Args:
        log_returns: Array of historical log returns.
        dt: Time step size (e.g., 1/252 for daily returns).
        S0: Initial stock price.
        n_simulations: Number of simulation paths.
        device: PyTorch device ('cuda' for GPU or 'cpu').

    Returns:
        avg_stock_prices: Average stock prices across simulations (n_days,).
        all_stock_prices: Simulated stock prices (n_days x n_simulations).
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)

    # Calculate the number of days to simulate
    n_days = int(1 / dt)
    # Convert log returns to NumPy array
    log_returns = np.array(log_returns)

    # Fit a Hidden Markov Model (HMM) to the log returns
    #model = GMMHMM(n_components=3, n_mix=2, random_state=42, covariance_type="diag")
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(log_returns.reshape(-1, 1))

    # Extract regime parameters
    means = model.means_.flatten()  # Mean log returns for each regime
    variances = np.sqrt(model.covars_).flatten()  # Standard deviations for each regime
    transition_matrix = model.transmat_  # Regime transition probabilities

    # Compute drift term
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    drift = mu - (sigma ** 2) / 2

    # Initialize PyTorch tensors
    simulated_log_returns = torch.zeros((n_days, n_simulations), dtype=torch.float32, device=device)

    # Simulate log returns for all paths in parallel
    regimes = torch.randint(0, 2, (n_simulations,), device=device)  # Initial regimes for each simulation

    for t in range(n_days):
        # Sample log returns for current regimes
        log_returns_t = torch.normal(
            mean=torch.tensor(means[regimes], device=device),
            std=torch.tensor(variances[regimes], device=device)
        ) + drift

        # Store log returns
        simulated_log_returns[t, :] = log_returns_t

        # Transition to next regimes
        transition_probs = torch.tensor(transition_matrix, device=device)
        regimes = torch.multinomial(transition_probs[regimes], num_samples=1).squeeze()

    # Compute cumulative log returns and convert to stock prices
    cumulative_log_returns = torch.cumsum(simulated_log_returns, dim=0)
    all_stock_prices = S0 * torch.exp(cumulative_log_returns)

    # Average prices over all simulation paths (columns)
    avg_stock_prices = torch.mean(all_stock_prices, dim=1)

    return avg_stock_prices.cpu().numpy(), all_stock_prices.cpu().numpy()



