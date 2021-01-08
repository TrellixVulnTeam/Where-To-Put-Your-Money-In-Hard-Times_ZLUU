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
    def __init__(self, stock_symbol, short_window=2, long_window=20, moving_avg='SMA', period='1y'):
        self.stock_symbol, self.short_window, self.long_window, self.moving_avg, self.period = stock_symbol, short_window, long_window, moving_avg, period

    def MovingAverageCrossStrategy(self, display_table=True):    
        stock_df = yf.download(self.stock_symbol, period=self.period)['Adj Close']
        stock_df = pd.DataFrame(stock_df) 
        stock_df.columns = {self.stock_symbol} 
        stock_df.dropna(axis = 0, inplace = True) 
        # column names for long and short moving average columns
        short_window_col = str(self.short_window) + '_' + self.moving_avg
        long_window_col = str(self.long_window) + '_' + self.moving_avg  
        if self.moving_avg == 'SMA':
            stock_df[short_window_col] = stock_df[self.stock_symbol].rolling(window = self.short_window, min_periods = 1).mean()
            stock_df[long_window_col] = stock_df[self.stock_symbol].rolling(window = self.long_window, min_periods = 1).mean()
        elif self.moving_avg == 'EMA':
            stock_df[short_window_col] = stock_df[self.stock_symbol].ewm(span = self.short_window, adjust = False).mean()
            stock_df[long_window_col] = stock_df[self.stock_symbol].ewm(span = self.long_window, adjust = False).mean()
        # create a new column 'Signal' such that if faster moving average is greater than slower moving average set Signal as 1 else 0.
        stock_df['Signal'] = 0.0  
        stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0) 
        # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
        stock_df['Position'] = stock_df['Signal'].diff()
        # plot close price, short-term and long-term moving averages
        plt.figure(figsize = (15,6), dpi = 100)
        plt.tick_params(axis = 'both', labelsize = 14)
        stock_df[self.stock_symbol].plot(color = 'k', lw = 3, label = self.stock_symbol)  
        stock_df[short_window_col].plot(color = 'b', lw = 2, label = short_window_col)
        stock_df[long_window_col].plot(color = 'g', lw = 2, label = long_window_col) 
        # plot 'buy' signals
        plt.plot(stock_df[stock_df['Position'] == 1].index, stock_df[
            short_window_col][stock_df['Position'] == 1],'^', markersize = 15, color = 'g', alpha = 0.7, label = 'buy')
        # plot 'sell' signals
        plt.plot(stock_df[stock_df['Position'] == -1].index, stock_df[
            short_window_col][stock_df['Position'] == -1], 'v', markersize = 15, color = 'r', alpha = 0.7, label = 'sell')
        plt.ylabel('Price in $ (USD)', fontsize = 16 )
        plt.xlabel('Date', fontsize = 16 )
        plt.title(str(self.stock_symbol) + ' - ' + str(self.moving_avg) + ' Crossover', fontsize = 20)
        plt.legend()
        plt.grid()
        if display_table == True:
            df_pos = stock_df[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)]
            df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
            print(f'\n                 ~ {self.moving_avg} Moving Average Model Signal Points ~')
            print(tabulate(df_pos, headers = 'keys', tablefmt = 'psql'))
            self.table_res = tabulate(df_pos, headers = 'keys', tablefmt = 'psql')
        return stock_df, self.table_res

# if __name__ == '__main__':
#     stock_df, table = MovingAverage3('AAPL', short_window=2, long_window=20, moving_avg='SMA', period='1y').MovingAverageCrossStrategy()
#     stock_df_ema, table_ema = MovingAverage3('GOOGL', short_window=2, long_window=20, moving_avg='EMA', period='1y').MovingAverageCrossStrategy()
#     plt.show()