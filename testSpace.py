import time
import torch
from torch.distributions import Normal, StudentT, LogNormal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm, t, lognorm
from joblib import Parallel, delayed
from fitter import Fitter
import csv
from scipy.stats import kstest
from scipy.stats import ks_2samp
from scipy.stats import norm, t, lognorm

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


# Optimized distribution fitting
def find_best_distributions(log_returns):
    distribution = ['norm', 'lognorm', 't']
    f = Fitter(log_returns, distributions=distribution)
    f.fit()
    best_fit = f.get_best()
    best_distribution = list(best_fit.keys())[0]
    best_params = best_fit[best_distribution]
    return best_distribution, best_params

SEED = 42
torch.manual_seed(SEED)  # For PyTorch
np.random.seed(SEED)

def simulate_markov_stock_prices_gpu(
    log_returns, dt, S0, best_n_components_bic, best_distribution, best_params, t_value, n_simulations, device=None
):
    """
    Simulates stock prices using a Markov regime-switching model on the GPU.

    Args:
        log_returns: Historical log returns.
        dt: Time step size (e.g., 1/252 for daily returns).
        S0: Initial stock price.
        best_n_components_bic: Optimal number of HMM components.
        best_distribution: Best-fit distribution name.
        best_params: Parameters of the best-fit distribution.
        t_value: T-distribution value for the current day.
        n_simulations: Number of simulations.
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        avg_stock_prices: Average simulated stock prices across paths.
        all_stock_prices: All simulated stock prices (n_simulations x n_days).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n_days = 1  # Simulating one day at a time
    log_returns = np.array(log_returns)

    # Fit Hidden Markov Model
    model = GaussianHMM(n_components=best_n_components_bic, covariance_type="diag", n_iter=500, random_state=SEED)
    model.fit(log_returns.reshape(-1, 1))
    means, variances = model.means_.flatten(), np.sqrt(model.covars_).flatten()
    transition_matrix = model.transmat_

    # Drift and scaling factors
    mu, sigma = np.mean(log_returns), np.std(log_returns)
    drift = mu - (sigma**2) / 2

    simulated_log_returns = torch.zeros((n_days, n_simulations), dtype=torch.float32, device=device)
    regimes = torch.randint(0, len(means), (n_simulations,), device=device)

    # Convert HMM parameters to tensors for GPU usage
    transition_probs = torch.tensor(transition_matrix, device=device)
    variances_tensor = torch.tensor(variances, device=device)
    means_tensor = torch.tensor(means, device=device)

    for time_step in range(n_days):
        samples = torch.zeros(n_simulations, device=device)
        for regime_idx in range(len(means)):
            sampled_regime_mask = (regimes == regime_idx)
            num_samples = sampled_regime_mask.sum().item()

            if num_samples > 0:
                # Generate samples from the best-fit distribution
                if best_distribution == "norm":
                    dist_samples = norm.rvs(
                        loc=best_params["loc"], scale=best_params["scale"], size=num_samples
                    )
                elif best_distribution == "t":
                    dist_samples = t.rvs(
                        df=best_params["df"], loc=best_params["loc"], scale=best_params["scale"], size=num_samples
                    )
                elif best_distribution == "lognorm":
                    dist_samples = lognorm.rvs(
                        s=best_params["s"], loc=best_params["loc"], scale=best_params["scale"], size=num_samples
                    )
                else:
                    raise ValueError(f"Unsupported distribution: {best_distribution}")

                # Convert distribution samples to tensors
                dist_samples = torch.tensor(dist_samples, dtype=torch.float32, device=device)
                samples[sampled_regime_mask] = dist_samples

        # Adjust samples based on HMM regime
        scaled_samples = samples * torch.sqrt(variances_tensor[regimes]) + means_tensor[regimes]
        scaled_samples += drift

        # Incorporate negative/positive t-values into log returns
        scaled_samples *= t_value  # t_value can be negative or positive
        simulated_log_returns[time_step, :] = scaled_samples

        # Update regimes based on transition probabilities
        regimes = torch.multinomial(transition_probs[regimes], num_samples=1).squeeze()

    # Compute cumulative log returns and convert to stock prices
    cumulative_log_returns = torch.cumsum(simulated_log_returns, dim=0)
    all_stock_prices = S0 * torch.exp(cumulative_log_returns)
    avg_stock_prices = torch.mean(all_stock_prices, dim=1)

    return avg_stock_prices.cpu().numpy(), all_stock_prices.cpu().numpy()




# Optimal component selection
def select_optimal_n_components(log_returns, max_components=10, covariance_type="diag", random_state=42):
    bic_values = []
    aic_values = []
    n_values = range(1, max_components + 1)

    log_returns = np.array(log_returns)
    log_returns = log_returns[~np.isnan(log_returns)]
    log_returns = log_returns[np.isfinite(log_returns)]

    if log_returns.size < 10:
        raise ValueError("Insufficient data size for training the HMM model.")

    scaler = StandardScaler()
    log_returns_scaled = scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()

    for n in n_values:
        try:
            model = GaussianHMM(n_components=n, covariance_type=covariance_type, n_iter=500, random_state=random_state)
            model.fit(log_returns_scaled.reshape(-1, 1))

            log_likelihood = model.score(log_returns_scaled.reshape(-1, 1))
            n_parameters = n * (n - 1) + 2 * n + 2 * n * 1
            data_points = log_returns_scaled.shape[0]

            bic = n_parameters * np.log(data_points) - 2 * log_likelihood
            aic = 2 * n_parameters - 2 * log_likelihood

            bic_values.append(bic)
            aic_values.append(aic)
        except ValueError:
            bic_values.append(np.inf)
            aic_values.append(np.inf)

    best_n_components_bic = n_values[np.argmin(bic_values)]
    best_n_components_aic = n_values[np.argmin(aic_values)]

    return best_n_components_bic, best_n_components_aic, bic_values, aic_values


def check_distribution(log_returns):
    distributions = ['norm', 'lognorm', 't']
    size = len(log_returns)  # Match the size of log_returns

    best_dist = ''  # Variable to store the best-fitting distribution
    best_statistic = float('inf')  # Initialize to largest possible value (infinity)

    for dist in distributions:
        if dist == 'norm':
            # Generate a normal distribution
            mean = np.mean(log_returns)
            std = np.std(log_returns)
            normal_values = np.random.normal(loc=mean, scale=std, size=size)
            statistic, p_value = ks_2samp(normal_values, log_returns)
            print('Normal Distribution - Statistic:', statistic, 'P-value:', p_value)

            if 0 <  p_value < best_statistic:
                best_dist = 'norm'
                best_statistic = p_value

        elif dist == 'lognorm':
            # Generate a lognormal distribution
            shape, loc, scale = 1.0, 0, np.mean(log_returns)  # Example parameters
            lognormal_values = np.random.lognormal(mean= scale, sigma=shape, size=size)
            statistic, p_value = ks_2samp(lognormal_values, log_returns)
            print('Lognormal Distribution - Statistic:', statistic, 'P-value:', p_value)

            if 0 < p_value < best_statistic:
                best_dist = 'lognorm'
                best_statistic = p_value

        elif dist == 't':
            # Generate a T-distribution
            degrees_of_freedom = 3
            t_values = np.random.standard_t(df=degrees_of_freedom, size=size) * 0.01
            statistic, p_value = ks_2samp(t_values, log_returns)
            print('T Distribution - Statistic:', statistic, 'P-value:', p_value)

            if 0 < p_value < best_statistic:
                best_dist = 't'
                best_statistic = p_value

    print('Best Distribution:', best_dist)

    if best_dist == 'norm':
        best_params = {'loc': np.mean(log_returns), 'scale': np.std(log_returns)}
    elif best_dist == 'lognorm':
        best_params = {'s': np.mean(log_returns), 'loc': 0, 'scale': np.std(log_returns)}
    elif best_dist == 't':
        df, loc, scale = t.fit(log_returns)
        best_params = {'df': df, 'loc': loc, 'scale': scale}
    return best_dist, best_params






if __name__ == "__main__":
    filesCsv = ['amazon.csv']
    names = [file.replace('.csv', '') for file in filesCsv]
    n_simulations = 10
    dt = 1 / 252
    start_time = time.time()

    dividends = [0.0] * len(filesCsv)
    train_data, test_data, data_dict, T, S0_dict = dict(filesCsv, dividends, 0.05, 20)

    for name in names:
        S0 = train_data[name]['S0']
        log_returns = train_data[name]['log_returns']
        test_prices = test_data[name]['prices']
        avg_stock_prices = []

        print(log_returns)
    """"
        # Find the best distribution and HMM parameters
        best_n_components_bic, _, _, _ = select_optimal_n_components(log_returns)
        best_distribution, best_params = find_best_distributions(log_returns)
        print(f"Best distribution for {name}: {best_distribution}, Parameters: {best_params}")

        # Generate T-distribution innovations
        t_values = np.random.standard_t(df=best_params["df"], size=252)
        S0_temp = S0
        t_sign = 0

        # Simulate for each day
        for k in range(252):
            t_sign = 1
            if t_values[k] < 0:
                t_sign = -1
            new_stock_prices, _ = simulate_markov_stock_prices_gpu(
                log_returns, dt, S0_temp, best_n_components_bic, best_distribution, best_params, t_sign, n_simulations
            )
            avg_stock_prices.append(new_stock_prices.mean())
            S0_temp = new_stock_prices[-1]

            if k % 15 == 0:  # Log progress every 15 days
                print(f"Completed {k / 5} weeks for {name}")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(avg_stock_prices, label="Predicted Prices")
        plt.plot(test_prices[:252], label="Actual Prices")
        plt.legend()
        plt.title(f"Price Prediction vs Actual Prices for {name}")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.show()
    """
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds")


