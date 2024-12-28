import time

from gbmAdjusted import bayesian_estimation_for_asset
from workSpace import dict
import torch
from fitter import Fitter
from scipy.stats import distributions
from arch import arch_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from workSpace import plotHist



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
        'norm', 'expon', 'gamma', 'beta', 'uniform',
        'lognorm', 't', 'pareto', 'cauchy', 'gumbel',
        'weibull', 'truncexpon', 'powerlaw', 'powerlognormal',
        'chi2', 'f', 'exponpow', 'laplace',
        'skewnorm', 'triang', 'logistic', 'hyperbolic',
        'rayleigh', 'nakagami', 'invgauss', 'gompertz',
        'loggamma', 'halfnorm', 'fatiguelife', 'exponnorm',
        'wald'
    ]
    f = Fitter(log_returns, distributions=distribution)
    f.fit()
    best_fit = f.get_best()
    best_distribution = list(best_fit.keys())[0]
    best_params = best_fit[best_distribution]
    return best_distribution, best_params



from scipy.stats import norm, t, lognorm


SEED = 42
torch.manual_seed(SEED)  # For PyTorch
np.random.seed(SEED)

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
    log_returns = np.array(log_returns)

    for n in n_values:
        model = GaussianHMM(n_components=n, covariance_type=covariance_type, n_iter=1000, random_state=random_state)
        model.fit(log_returns.reshape(-1, 1))

        log_likelihood = model.score(log_returns.reshape(-1, 1))  # Log-likelihood of the model
        n_parameters = n * (n - 1) + 2 * n + 2 * n * 1  # Approximation of parameters
        data_points = log_returns.shape[0]

        # Compute BIC and AIC
        bic = n_parameters * np.log(data_points) - 2 * log_likelihood
        aic = 2 * n_parameters - 2 * log_likelihood

        bic_values.append(bic)
        aic_values.append(aic)

    # Find the best n_components
    best_n_components_bic = n_values[np.argmin(bic_values)]
    best_n_components_aic = n_values[np.argmin(aic_values)]

    return best_n_components_bic, best_n_components_aic, bic_values, aic_values




if __name__ == "__main__":
    filesCsv = ['ko.csv']
    name = filesCsv[0].split('.')[0]
    n_simulations = 2000
    start_time = time.time()

    dividends = [0.0]
    if len(filesCsv) != len(dividends):
        raise ValueError("Number of files and dividends must be equal")

    train_data, test_data, data_dict, T, S0_dict = dict(filesCsv, dividends, 0.05, 20)
    log_returns = train_data[name]['log_returns']
    S0 = train_data[name]['S0']
    print(S0)
    dt = 1 / 252
    best_n_components_bic, best_n_components_aic, bic_values, aic_values = select_optimal_n_components(log_returns)
    #print(best_n_components_bic, best_n_components_aic)
    avg_stock_prices, all_stock_prices = simulate_markov_stock_prices_gpu(log_returns, dt, S0, best_n_components_bic, n_simulations)
    print(avg_stock_prices)
    print(test_data[name]['final'])

    elapsed_time = round(time.time() - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds")





