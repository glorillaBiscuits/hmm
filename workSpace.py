import time
import numpy as np
from statistics import mean, variance, stdev
from priceSimulator import csvRead
from gbmAdjusted import estimate_bayesian_parameters
from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf
from arch import arch_model
import matplotlib.pyplot as plt
import torch
from fitter import Fitter

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


def hrs_single_parallel_alt(log_returns, n_simulations, T, dt, S0, dist):
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
    #np.random.seed(42)  # Ensures consistent random numbers
    #tf.random.set_seed(42)
    # Simulation parameters
    n_steps = int(T / dt)
    S0 = float(S0)  # Ensure S0 is scalar

    # Convert log returns to NumPy array for processing
    log_returns = np.array(log_returns, dtype=np.float32)

    # Rescale log returns for GARCH model fitting
    scale_factor = 100
    scaled_returns = log_returns * scale_factor

    # Fit GARCH(1,1) model to the rescaled log returns
    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist= dist)
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
    print(f"Predicted Value: {mean_daily_prices_np[-1]}")
    return mean_daily_prices_np.tolist()

def plotHist(log_returns, title):
    # Create a histogram of log_returns
    plt.figure(figsize=(8, 6))
    plt.hist(log_returns, bins=20, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Log Return Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def findDist(log_returns):
    distributions = [
        'norm', 'expon', 'gamma', 'beta', 'uniform',
        'lognorm', 't', 'pareto', 'cauchy', 'gumbel',
        'weibull', 'truncexpon', 'powerlaw', 'powerlognormal',
        'chi2', 'f', 'exponpow', 'laplace',
        'skewnorm', 'triang', 'logistic', 'hyperbolic',
        'rayleigh', 'nakagami', 'invgauss', 'gompertz',
        'loggamma', 'halfnorm', 'fatiguelife', 'exponnorm',
        'wald'
        ]

    f = Fitter(log_returns, distributions=distributions)
    f.fit()
    f.summary()
    best_fit = f.get_best()
    best_distribution = list(best_fit.keys())[0]
    best_params = best_fit[best_distribution]
    print(f"The best fit is the '{best_distribution}' distribution with parameters: {best_params}")
    return best_distribution

def simulate_garch_paths_mps(log_returns, T, dt, S0, distr, n_simulations, scale_factor, device=None):
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
    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist= distr)
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

def overlay_best_fit(all_simulated_returns, best_fit_distribution, Mu, Sigma):
    """
    Plots a histogram of simulated returns with the best-fit distribution overlayed.

    Args:
    - all_simulated_returns: Flattened array of simulated log returns.
    - best_fit_distribution: String representing the name of the distribution (e.g., 'norm', 'laplace').
    - params: Tuple of parameters for the distribution.

    Returns:
    - None: Displays the plot.
    """
    from scipy.stats import distributions

    # Flatten simulated returns
    flattened_returns = all_simulated_returns.flatten()

    # Plot histogram of simulated returns
    plt.figure(figsize=(8, 6))
    plt.hist(flattened_returns, bins=20, density=True, edgecolor='black', alpha=0.7, label='Simulated Returns')

    # Generate x-values for the PDF
    x = np.linspace(min(flattened_returns), max(flattened_returns), 1000)

    # Dynamically get the distribution from scipy.stats
    try:
        dist = getattr(distributions, best_fit_distribution)
    except AttributeError:
        print(f"Unsupported distribution: {best_fit_distribution}")
        return

    # Compute the PDF using the provided parameters
    pdf = dist.pdf(x, Mu, Sigma)

    # Overlay the PDF
    plt.plot(x, pdf, label=f'{best_fit_distribution} Fit', linewidth=2)

    # Add labels and title
    plt.title("Simulated Returns with Best-Fit Distribution")
    plt.xlabel("Log Return Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()



if __name__ == "__main__":
    #asset data
    filesCsv = [
        'apple.csv'
        #'amazon.csv',
        #'asml.csv'
    ]

    name = filesCsv[0].split('.')[0]

    start_time = time.time()

    #corresponding dividend return in as decimal
    dividends = [0.0]
    # dividends must correspond to the number of assets
    if len(filesCsv) != len(dividends):
        raise ValueError("Number of files and dividends must be equal")

    """
    dictionary with names as keys for assets and files for csv file of data 
    """

    window = 0.05# % of the data for testing
    recent = 20

    train_data, test_data, data_dict, T, S0_dict = dict(filesCsv, dividends, window, recent)


    n_simulations = 30000  # 1000000
    n_paths = 1000  # 100000
    # limit the optimized portfolio volatility to this
    maxVolatility = 0.15
    dt = 1 / 252  # single trading day
    scale_factor = 100
    test_returns = test_data[name]['log_returns']
    log_returns = train_data[name]['log_returns']
    Mu = mean(train_data[name]['log_returns'])
    Sigma = stdev(train_data[name]['log_returns'])
    S0 = train_data[name]['S0']
    #distr = findDist(test_returns)
    distr = 't'
    n_log = np.array(test_returns)
    f_log = n_log.flatten()
    overlay_best_fit(np.array(log_returns), distr, Mu, Sigma)
    #w = hrs_single_parallel_alt(log_returns, n_simulations, T, dt, S0, distr)
    k = simulate_garch_paths_mps(log_returns, T, dt, S0, distr, n_simulations, scale_factor)
    print('Predicted Price: ', k[-1])
    print('Real Price: ', test_data[name]['final'])
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds")
