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
from priceSimulator import dict
from scipy.stats import kstest
from scipy.stats import ks_2samp

SEED = 42
torch.manual_seed(SEED)  # For PyTorch
np.random.seed(SEED)


# Optimized distribution fitting
def find_best_distributions(log_returns):
    distribution = ['norm', 'lognorm', 't']
    f = Fitter(log_returns, distributions=distribution)
    f.fit()
    best_fit = f.get_best()
    best_distribution = list(best_fit.keys())[0]
    best_params = best_fit[best_distribution]
    return best_distribution, best_params

# Optimized simulation function
def simulate_markov_stock_prices_gpu(log_returns, dt, S0, best_n_components_bic, n_simulations, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)

    n_days = int(1 / dt)
    log_returns = np.array(log_returns)

    model = GaussianHMM(n_components=best_n_components_bic, covariance_type="diag", n_iter=500, random_state=SEED)
    model.fit(log_returns.reshape(-1, 1))

    means = model.means_.flatten()
    variances = np.sqrt(model.covars_).flatten()
    transition_matrix = model.transmat_

    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    drift = mu - (sigma ** 2) / 2

    best_distribution, best_params = find_best_distributions(log_returns)
    print(f"Best distribution: {best_distribution}, Parameters: {best_params}")
    #best_distribution2, best_params2 = find_best_distributions(log_returns)
    #print(f"Best distribution: {best_distribution2}, Parameters: {best_params2}")

    simulated_log_returns = torch.zeros((n_days, n_simulations), dtype=torch.float32, device=device)
    regimes = torch.randint(0, len(means), (n_simulations,), device=device)

    transition_probs = torch.tensor(transition_matrix, device=device)
    variances_tensor = torch.tensor(variances, device=device)
    means_tensor = torch.tensor(means, device=device)

    for time_step in range(n_days):
        samples = torch.zeros(n_simulations, device=device)
        for regime_idx in range(len(means)):
            sampled_regime_mask = (regimes == regime_idx)
            num_samples = sampled_regime_mask.sum().item()

            if num_samples > 0:
                if best_distribution == "norm":
                    dist_samples = norm.rvs(loc=best_params["loc"], scale=best_params["scale"], size=num_samples)
                elif best_distribution == "t":
                    dist_samples = t.rvs(df=best_params["df"], loc=best_params["loc"], scale=best_params["scale"], size=num_samples)
                elif best_distribution == "lognorm":
                    dist_samples = lognorm.rvs(s=best_params["s"], loc=best_params["loc"], scale=best_params["scale"], size=num_samples)
                dist_samples = torch.tensor(dist_samples, dtype=torch.float32, device=device)
                samples[sampled_regime_mask] = dist_samples

        scaled_samples = samples * torch.sqrt(variances_tensor[regimes]) + means_tensor[regimes]
        scaled_samples += drift
        simulated_log_returns[time_step, :] = scaled_samples

        regimes = torch.multinomial(transition_probs[regimes], num_samples=1).squeeze()

    cumulative_log_returns = torch.cumsum(simulated_log_returns, dim=0)
    all_stock_prices = S0 * torch.exp(cumulative_log_returns)
    avg_stock_prices = torch.mean(all_stock_prices, dim=1)

    return avg_stock_prices.cpu().numpy(), all_stock_prices.cpu().numpy()

# Parallelized file processing
def process_file(name, train_data, test_data, dt, n_simulations):
    log_returns = train_data[name]['log_returns']
    S0 = train_data[name]['S0']
    test_prices = test_data[name]['prices']

    best_n_components_bic, _, _, _ = select_optimal_n_components(log_returns)
    avg_stock_prices, _ = simulate_markov_stock_prices_gpu(log_returns, dt, S0, best_n_components_bic, n_simulations)

    plt.plot(avg_stock_prices, label="Predicted Prices")
    plt.plot(test_prices, label="Actual Prices")
    plt.legend()
    plt.title(f"Price Prediction vs Actual Prices for {name}")
    plt.show()

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
    filesCsv = ['amazon.csv', 'apple.csv', 'asml.csv', 'googl.csv', 'ko.csv', 'nvidia.csv']
    names = [file.replace('.csv', '') for file in filesCsv]
    n_simulations = 2000
    dt = 1 / 252
    start_time = time.time()

    dividends = [0.0] * len(filesCsv)
    if len(filesCsv) != len(dividends):
        raise ValueError("Number of files and dividends must be equal")

    train_data, test_data, data_dict, T, S0_dict = dict(filesCsv, dividends, 0.05, 20)

    Parallel(n_jobs=-1)(
        delayed(process_file)(name, train_data, test_data, dt, n_simulations) for name in names
    )
    log_returns = train_data['amazon']['log_returns']
    check_distribution(log_returns)
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds")