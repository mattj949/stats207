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
    
    portfolio_value = 1
    long_only_value = 1
    list_of_portfolio_values = []
    list_of_long_only_values = []
    
    # Scale our holding size by the inverse of the asset's price volatility
    pd_series = pd.Series(price_series)
    vol = pd_series.pct_change().std()
    scaling = 0.05/vol # Target 5% volatility


    list_of_weights = []

    print(scaling)
    
    for i in range(1, len(price_series)):
        # Look back a day and see whether we went long or short the asset
        
        # If we forecasted a higher price for today than the price yesterday, we go long:
        if forecast_for_each_day[i] >= price_series[i-1]: 
            long_short = 1
        else:
            long_short = -1
        
        list_of_weights += [long_short]

        print("Predicted price change: ")
        print("   " + str(forecast_for_each_day[i] - price_series[i-1]))
        print("Actual price change:")
        print("   " + str(price_series[i] - price_series[i-1]))



        asset_return = ((price_series[i] - price_series[i-1]) / price_series[i-1])
        holding_return = long_short * scaling * asset_return
        
        portfolio_value = portfolio_value * (1+holding_return)
        long_only_value = long_only_value * (1+asset_return)
        
        list_of_portfolio_values += [portfolio_value]
        list_of_long_only_values += [long_only_value]
        

    plt.figure(figsize=(18,9))
    plt.plot(list_of_portfolio_values)
    plt.plot(list_of_long_only_values)
    plt.title("Backtest vs. long only")
    plt.legend(["Strategy return", "Long-Only Return"])
    plt.show()
    


    plt.figure(figsize=(18,9))
    plt.plot(np.asarray(forecast_for_each_day) - np.asarray(price_series))
    plt.plot(np.asarray(list_of_weights) / 5)
    plt.show()