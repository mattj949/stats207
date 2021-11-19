from matplotlib import pyplot as plt
import pandas as pd

# Start off simple, buy if prediction says price will go up, otherwise sell.
# Input: one_step_ahead - the price forecasts for the next day [a vector across time]
# input: price_series   - the actual historical price series

# forecast[i] should be the forecast for the price on day i, as of day i-1
# --> "Yesterday's forecast for today"
# price_series[i] should be the actual price on day i
def backtest(forecast_for_each_day, price_series):
    
    portfolio_value = 1
    list_of_portfolio_values = []
    
    # Scale our holding size by the inverse of the asset's price volatility
    pd_series = pd.Series(price_series)
    vol = pd_series.pct_change().std()
    scaling = 1/vol
    
    for i in range(1, len(price_series)):
        # Look back a day and see whether we went long or short the asset
        holding_quantity = [1 if one_step_ahead[i-1] >= price_series[i-1] else -1]
        
        holding_return = holding_quantity * ((price_series[i] - price_series[i-1]) / price_series[i-1])
        
        portfolio_value = portfolio_value * (scaling*(1+holding_return))
        
        list_of_portfolio_values += [portfolio_value]
        
        
    plt.plot(list_of_portfolio_values)
    plt.title("Backtest")
    plt.show()