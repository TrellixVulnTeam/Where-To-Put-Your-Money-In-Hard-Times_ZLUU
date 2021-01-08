import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
import pandas_datareader.data as web
import yfinance as yf

class MovingAverage3(object):
    def __init__(self):
        pass

    def MovingAverageCrossStrategy(
        self,stock_symbol = '^GSPC',
        short_window = 2,
        long_window = 20,
        moving_avg = 'SMA',
        period = '1y',
        display_table = True
        ):
        stock_df = yf.download(stock_symbol, period=period, interval='1d')['Adj Close']
        stock_df = pd.DataFrame(stock_df) # convert Series object to dataframe 
        stock_df.columns = {stock_symbol} # assign new colun name
        stock_df.dropna(axis = 0, inplace = True) # remove any null rows 
        # column names for long and short moving average columns
        short_window_col = str(short_window) + '_' + moving_avg
        long_window_col = str(long_window) + '_' + moving_avg  
        if moving_avg == 'SMA':
            stock_df[short_window_col] = stock_df[stock_symbol].rolling(window = short_window, min_periods = 1).mean()
            stock_df[long_window_col] = stock_df[stock_symbol].rolling(window = long_window, min_periods = 1).mean()
        elif moving_avg == 'EMA':
            stock_df[short_window_col] = stock_df[stock_symbol].ewm(span = short_window, adjust = False).mean()
            stock_df[long_window_col] = stock_df[stock_symbol].ewm(span = long_window, adjust = False).mean()
        # create a new column 'Signal' such that if faster moving average is greater than slower moving average set Signal as 1 else 0.
        stock_df['Signal'] = 0.0  
        stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0) 
        # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
        stock_df['Position'] = stock_df['Signal'].diff()
        # plot close price, short-term and long-term moving averages
        plt.figure(figsize = (15,6), dpi = 100)
        plt.tick_params(axis = 'both', labelsize = 14)
        stock_df[stock_symbol].plot(color = 'k', lw = 3, label = stock_symbol)  
        stock_df[short_window_col].plot(color = 'b', lw = 2, label = short_window_col)
        stock_df[long_window_col].plot(color = 'g', lw = 2, label = long_window_col) 
        # plot 'buy' signals
        plt.plot(stock_df[stock_df['Position'] == 1].index, stock_df[
            short_window_col][stock_df['Position'] == 1],'^', markersize = 15, color = 'g', alpha = 0.7, label = 'buy')
        # plot 'sell' signals
        plt.plot(stock_df[stock_df['Position'] == -1].index, stock_df[
            short_window_col][stock_df['Position'] == -1], 'v', markersize = 15, color = 'r', alpha = 0.7, label = 'sell')
        plt.ylabel('Price in ₹', fontsize = 16 )
        plt.xlabel('Date', fontsize = 16 )
        plt.title(str(stock_symbol) + ' - ' + str(moving_avg) + ' Crossover', fontsize = 20)
        plt.legend()
        plt.grid()
        if display_table == True:
            df_pos = stock_df[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)]
            df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
            print(tabulate(df_pos, headers = 'keys', tablefmt = 'psql'))
            self.res = tabulate(df_pos, headers = 'keys', tablefmt = 'psql')
        return (stock_df, self.res)

if __name__ == '__main__':
    x = MovingAverage3()
    stock_df, table = x.MovingAverageCrossStrategy('AAPL', 20, 50, 'SMA', '1y')
    # stock_df, table = x.MovingAverageCrossStrategy('AAPL', 20, 50, 'EMA', '1y')
    plt.show()