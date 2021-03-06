from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Start off simple, buy if prediction says price will go up, otherwise sell.
# Input: one_step_ahead - the price forecasts for the next day [a vector across time]
# input: price_series   - the actual historical price series

# forecast[i] should be the forecast for the price on day i, as of day i-1
# --> "Yesterday's forecast for today"
# price_series[i] should be the actual price on day i
def backtest(forecast_for_each_day, price_series, reference_series):
    
    
    sim = pd.Series(forecast_for_each_day)
    true = pd.Series(price_series)

    sim_pct = sim.pct_change()
    true_pct = true.pct_change()

    total = len(price_series)
    
    num_correct = 0
    for i in range(total):
        if sim_pct[i] > 0 and true_pct[i] > 0:
            num_correct += 1
        elif sim_pct[i] < 0 and true_pct[i] < 0:
            num_correct += 1

    print("Percentage of time our forecast was correct: " + str(num_correct / total))
    
    
    
    
    sim = pd.Series(reference_series)
    true = pd.Series(price_series)

    sim_pct = sim.pct_change()
    true_pct = true.pct_change()
    
    num_correct = 0
    for i in range(total):
        if sim_pct[i] > 0 and true_pct[i] > 0:
            num_correct += 1
        elif sim_pct[i] < 0 and true_pct[i] < 0:
            num_correct += 1

    print("Percentage of time reference forecast was correct: " + str(num_correct / total))
    
    
    # Previous observations to use in calculating moving average price in refrence momentum strategy
    # I used 8 since I provided weekly data - 8 units = 8 weeks = ~2 months.
    REFERENCE_LOOKBACK = 8
    
    
    strategy_portfolio_value = 1
    reference_portfolio_value = 1
    list_strategy_portfolio_values = []
    list_reference_portfolio_values = []
    
    # Scale our holding size by the inverse of the asset's price volatility
    pd_series = pd.Series(price_series)
    vol = pd_series.pct_change().std()
    scaling = 0.05/vol # Target 15% volatility


    list_of_weights = []
    list_of_strategy_returns = []
    list_of_reference_returns = []
    
    print("Scaling: ", scaling)
    
    for i in range(1, len(price_series)):
        # Look back a day and see whether we went long or short the asset
        
        # If we forecasted a higher price than yesterday's price, we should have bought the asset.
        if forecast_for_each_day[i] >= price_series[i-1]: 
            long_short = 1
        else:
            long_short = -1
        
        list_of_weights += [long_short]
    
    
        moving_average = np.average(price_series[max(0,i-REFERENCE_LOOKBACK):i])
        
        if moving_average >= price_series[i-1]: 
            reference_long_short = 1
        else:
            reference_long_short = -1
        
        asset_return = ((price_series[i] - price_series[i-1]) / price_series[i-1])
        
        scaled_strategy_return = long_short * scaling * asset_return
        scaled_reference_return = reference_long_short * scaling * asset_return
        
        list_of_strategy_returns += [scaled_strategy_return]
        list_of_reference_returns += [scaled_reference_return]
        
        strategy_portfolio_value = strategy_portfolio_value * (1+scaled_strategy_return)
        reference_portfolio_value = reference_portfolio_value * (1+scaled_reference_return)
        
        list_strategy_portfolio_values += [strategy_portfolio_value]
        list_reference_portfolio_values += [reference_portfolio_value]
        
        
    print("Strategy Sharpe: " + str((np.mean(list_of_strategy_returns) / np.std(list_of_strategy_returns)) * np.sqrt(252)))
    print("Reference Sharpe: " + str((np.mean(list_of_reference_returns) / np.std(list_of_reference_returns)) * np.sqrt(252)))
          

    plt.figure(figsize=(18,9))
    plt.plot(list_reference_portfolio_values)
    plt.plot(list_strategy_portfolio_values)
    plt.title("Reference Strategy vs. ARIMA-based Strategy")
    plt.legend(["Reference Strategy", "ARIMA Strategy"])
    plt.show()
    


    plt.figure(figsize=(18,9))
    plt.plot(np.asarray(forecast_for_each_day) - np.asarray(price_series))
    plt.plot(np.asarray(list_of_weights) / 5)
    plt.legend(["Forecast - actual", "Long or Short"])
    plt.show()

    return list_strategy_portfolio_values, list_reference_portfolio_values

def backtest_log(predicted_log_ret, actual_log_ret):
    
    strategy_portfolio_value = 1
    long_only_portfolio_value = 1

    list_strategy_portfolio_values = []
    list_long_only_portfolio_values = []

    strategy_returns = []
    long_only_returns = []
    strategy_predictors = []
    correctdirections = []

    for i in range(min(len(actual_log_ret), len(predicted_log_ret))):
        # Scale by minmax of previous realized log returns
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
        # up to and including current predicted return - better the more data you have
        # squishes signals into 0, 1 range

        # if i == 0:
        #     scaling = 1
        # else:
        #     lookback = np.abs(predicted_log_ret[max(0, i - 20):i+1])
        #     v_scaled = min_max_scaler.fit_transform(lookback.reshape(-1, 1))
        #     scaling = v_scaled[-1]
        # assert(0 <= scaling <= 1.01)
        # scaling = float(scaling*np.sign(predicted_log_ret[i]))

        #strategy_predictors.append(scaling)
        strategy_predictors.append(predicted_log_ret[i])
        correctdirections.append(int(np.sign(predicted_log_ret[i]) == np.sign(actual_log_ret[i])))

        #strategy_returns.append(scaling * actual_log_ret[i])
        strategy_returns.append(predicted_log_ret[i]*actual_log_ret[i])

        long_only_returns.append(actual_log_ret[i])


        #strategy_portfolio_value = strategy_portfolio_value * np.exp(scaling*actual_log_ret[i])
        strategy_portfolio_value = strategy_portfolio_value * np.exp(strategy_returns[-1])

        long_only_portfolio_value = long_only_portfolio_value * np.exp(long_only_returns[-1])
        
        list_strategy_portfolio_values += [strategy_portfolio_value]
        list_long_only_portfolio_values += [long_only_portfolio_value]
    
    # psuedo sharpe
    print("Long only annualized psuedo sharpe: ", np.mean(long_only_returns)/np.std(long_only_returns)*np.sqrt(252))
    print("Strategy annualized psuedo sharpe: ", np.mean(strategy_returns)/np.std(strategy_returns)*np.sqrt(252))
    print("Correlation matrix between strategy predictors and returns")
    print(np.corrcoef(strategy_predictors, long_only_returns))
    print("Avg. Portfolio Allocation, ", np.mean(np.abs(strategy_predictors)))

    print("Percent of time predicted direction correctly: ", np.sum(correctdirections)/len(correctdirections)* 100)
    print("Percent of time long: ", np.sum(actual_log_ret > 0)/len(correctdirections)*100)

    plt.figure(figsize=(18,9))
    plt.plot(np.cumsum(np.array(strategy_returns)))
    #plt.plot(np.cumsum(np.array(long_only_returns)))
    plt.title("Model Strategy Returns")
    plt.legend(["Strategy return", "Long-Only Return"])
    plt.show()

    plt.figure(figsize=(18,9))
    plt.plot(strategy_predictors)
    plt.title("Signals")
    plt.show()

    plt.figure(figsize=(18,9))
    plt.plot(predicted_log_ret)
    plt.plot(actual_log_ret)
    plt.title("Actual vs. Predicted")
    plt.legend(["Predicted Log Return", "Actual Log Return"])
    plt.show()

    return strategy_returns, long_only_returns, strategy_predictors
