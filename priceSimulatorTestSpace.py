import time
import csv
import math
import pandas as pd
from statistics import mean, variance
from math import sqrt
from priceSimulator import csvRead
from priceSimulator import evaluate_metrics
from priceSimulator import estimate_bayesian_parameters
from priceSimulator import jump_diffusion_multiple_assets
from priceSimulator import historicalReturnsSample_t
from priceSimulator import gbmPrice_parallel
import tensorflow as tf
from scipy.stats import skew, kurtosis, gaussian_kde

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
    S0 = []
    S0_dict = {}
    shortData = {}
    s0 = []
    start = []
    names = []
    # Train-test split with 2 years for testing
    train_data = {}
    test_data = {}
    for name in csvDict['names']:
        S0_dict[name] = data_dict[name]['prices'][-1]
        S0.append(data_dict[name]['prices'][-1])
        names.append(name)
        # shortList = data_dict[name]['prices'][:-252]
        shortList = data_dict[name]['prices'][:-504]
        start.append(shortList[-1])
        s0.append(shortList[-1])
        # shortLog = data_dict[name]['log_returns'][:-252]
        shortLog = data_dict[name]['log_returns'][:-504]
        shortData[name] = {
            'prices': shortList,
            'log_returns': shortLog
        }


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

    #touple - name, dictionary - metric : array
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
    print(simulated_prices)
    """
    hrs = historicalReturnsSample_t(train_data, names, T, dt, S0, n_simulations, history_length=50, convergence_threshold=1e-6)
    hours = hrs[1]
    print("Bayesian Parameters Estimation + Jump Diffusion")
    for name in shortData.keys():
        evaluate_metrics(test_data[name]['log_returns'], simulated_prices[name], name)
        print("\n\nHistorical Returns Sample")
        evaluate_metrics(test_data[name]['log_returns'], hours[name], name)
        print("\n\n")
    """