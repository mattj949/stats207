from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Start off simple, buy if prediction says price will go up, otherwise sell.
# Input: one_step_ahead - the price forecasts for the next day [a vector across time]
# input: price_series   - the actual historical price series

# forecast[i] should be the forecast for the price on day i, as of day i-1
# --> "Yesterday's forecast for today"
# price_series[i] should be the actual price on day i
def backtest(forecast_for_each_day, price_series):
    
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
    scaling = 0.15/vol # Target 15% volatility


    list_of_weights = []

    print(scaling)
    
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
        
        strategy_portfolio_value = strategy_portfolio_value * (1+scaled_strategy_return)
        reference_portfolio_value = reference_portfolio_value * (1+scaled_reference_return)
        
        list_strategy_portfolio_values += [strategy_portfolio_value]
        list_reference_portfolio_values += [reference_portfolio_value]
        

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