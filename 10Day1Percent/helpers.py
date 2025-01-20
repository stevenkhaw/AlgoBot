# region imports
from AlgorithmImports import *
from scipy.stats import linregress
# endregion

def get_one_day_features(history_close_ten_day):

    # Initialize variables
    window_size = 5
    num_windows = len(history_close_ten_day) - window_size + 1

    results = []

    # Loop through overlapping windows
    for i in range(num_windows):
        prices = history_close_ten_day[i:i+window_size]
        idx = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = linregress(idx, prices)
        
        # Store the results as a tuple
        results.append({
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'std_err': std_err,
            'window_start': i
        })

    last_slope = results[-1]['slope']

    # Sort results based on a criterion (e.g., smallest standard error)
    results.sort(key=lambda x: x['std_err'])

    # Extract best and second best
    best = results[0]
    sec_best = results[1]

    # Now calculate close_minus_best_intercept and close_minus_sec_best_intercept
    current_close = history_close_ten_day.iloc[-1]

    close_minus_best_intercept = current_close - best['intercept']
    close_minus_sec_best_intercept = current_close - sec_best['intercept']

    best_slope = best['slope']
    best_intercept = best['intercept']
    best_se = best['std_err']
    best_t = best['r_value']
    best_idx = best['window_start'] + np.argmax(history_close_ten_day[best['window_start']:best['window_start'] + window_size])

    sec_best_slope = sec_best['slope']
    sec_best_intercept = sec_best['intercept']
    sec_best_se = sec_best['std_err']
    sec_best_t = sec_best['r_value']
    sec_best_idx = sec_best['window_start'] + np.argmax(history_close_ten_day[sec_best['window_start']:sec_best['window_start'] + window_size])

    std = history_close_ten_day.std()

    output = {
        'std': std,
        'slope': last_slope,
        'best_slope': best_slope,
        'best_intercept': best_intercept,
        'best_se': best_se,
        'best_t': best_t,
        'best_idx': best_idx,
        'sec_best_slope': sec_best_slope,
        'sec_best_intercept': sec_best_intercept,
        'sec_best_se': sec_best_se,
        'sec_best_t': sec_best_t,
        'sec_best_idx': sec_best_idx}

    return output



import pandas as pd
import numpy as np
from bayesian_linear_regression import batch_bocd

def compute_features_from_daily_data(daily_data, max_len=75, hazard=1e-2, batch_size=500):
    """
    Computes feature values using BOCD from daily ETF data.

    Parameters:
    - daily_data (pd.DataFrame or np.ndarray): DataFrame or NumPy array containing daily ETF data with 'Close' column.
    - max_len (int): Maximum length of the data segment for BOCD.
    - hazard (float): Hazard rate for BOCD.
    - batch_size (int): Batch size for BOCD.

    Returns:
    - pd.DataFrame: DataFrame containing the computed features.
    """
    # Convert DataFrame to NumPy array if necessary
    if isinstance(daily_data, pd.DataFrame):
        daily_data = daily_data['close'].to_numpy()
    
    # Check if the data is sufficient
    if len(daily_data) <= max_len:
        raise ValueError("The input data must have more than `max_len` rows")

    # Apply BOCD to the 'Close' prices
    R, result = batch_bocd(np.log(daily_data), max_len=max_len, hazard=hazard, batch_size=batch_size)

    # Compute argsort to find the best and second best change points
    argsort = np.argsort(R, axis=1)
    best = result[np.arange(result.shape[0]), argsort[:, -1]]
    sec_best = result[np.arange(result.shape[0]), argsort[:, -2]]

    # Use indices from the end of the data
    log_prices = np.log(daily_data)
    close_minus_best_intercept = log_prices[-best.shape[0]:] - best['intercept']
    close_minus_sec_best_intercept = log_prices[-best.shape[0]:] - sec_best['intercept']

    # Compute standard deviation and slope
    std = np.std(daily_data)
    slopes = np.gradient(daily_data)

    # Adjust shapes for concatenation
    n_samples = best.shape[0]
    
    best_slope = best['slope'][:, np.newaxis]
    best_idx = argsort[:, -1][:, np.newaxis]
    best_t = best['t'][:, np.newaxis]
    best_mean = best['mean'][:, np.newaxis]
    close_minus_best_intercept = close_minus_best_intercept[:, np.newaxis]
    best_se = best['se'][:, np.newaxis]

    sec_best_slope = sec_best['slope'][:, np.newaxis]
    sec_best_idx = argsort[:, -2][:, np.newaxis]
    sec_best_t = sec_best['t'][:, np.newaxis]
    sec_best_mean = sec_best['mean'][:, np.newaxis]
    close_minus_sec_best_intercept = close_minus_sec_best_intercept[:, np.newaxis]
    sec_best_se = sec_best['se'][:, np.newaxis]

    mean_slope = np.mean(slopes[-n_samples:])  # Adjusted mean slope calculation
    
    # Create the feature DataFrame
    features = pd.DataFrame(np.concatenate((
                                 best_slope,
                                 best_idx,
                                 best_t,
                                 #best_mean,
                                 close_minus_best_intercept,
                                 best_se,

                                 sec_best_slope,
                                 sec_best_idx,
                                 sec_best_t,
                                 #sec_best_mean,
                                 close_minus_sec_best_intercept,
                                 sec_best_se,

                                 np.full((n_samples, 1), mean_slope),
                                 np.full((n_samples, 1), std)
                               ),
                               axis=1),
                columns=('best_slope', 'best_idx', 'best_t', #'best_mean',
                         'close_minus_best_intercept', 'best_se',
                         'sec_best_slope', 'sec_best_idx', 'sec_best_t', #'sec_best_mean',
                         'close_minus_sec_best_intercept', 'sec_best_se',
                         'slope', 'std'))
    return features


def select_option_contract(self, data, prediction):

    # Get available option contracts
    option_chain = data.OptionChains.get(self.option.Symbol)

    if option_chain is None:
        return None

    # Pick the most liquid contract (ATM or slightly OTM)
    # If the prediction is True (outside ±1%), expect volatility (buy call or put based on direction)
    # If False, expect stability (do nothing or maybe sell option if you already hold one)
    
    # Get current SPY price
    spot_price = self.Securities[self.symbol].Price

    # Predict volatility, we'll choose an ATM contract
    contracts = sorted(option_chain, key=lambda x: abs(x.Strike - spot_price))
    
    if len(contracts) == 0:
        return None

    # True: price expected to move outside ±1% range
    if prediction:  
        
        # Buy both an ATM call and ATM put
        atm_call = next((contract for contract in contracts if contract.Right == OptionRight.Call), None)
        atm_put = next((contract for contract in contracts if contract.Right == OptionRight.Put), None)

        if atm_call is None or atm_put is None:
            return None
        
        # Buy one call option
        self.MarketOrder(atm_call.Symbol, 1)

        # Buy one put option
        self.MarketOrder(atm_put.Symbol, 1)
        
        return atm_call.Symbol, atm_put.Symbol
        
    # False: price expected to remain within ±1% range
    else:
        return None