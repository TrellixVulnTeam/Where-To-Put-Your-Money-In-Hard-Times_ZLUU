p = '/home/gordon/work/assemble/data/raw/'
import pickle
# from models.script.make_dataset1 import Stock_Info
import pandas as pd
import numpy as np
from pandas.io.pickle import read_pickle
from yahoo_fin.stock_info import *
import warnings
warnings.filterwarnings('ignore')
# from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set_palette('cubehelix')
# plt.style.use('seaborn-colorblind') #alternative
sns.set(style='darkgrid', context='talk', palette='Dark2')
sm, med, lg = 10, 15, 25
plt.rc('font', size = sm)         # controls default text sizes
plt.rc('axes', titlesize = med)   # fontsize of the axes title
plt.rc('axes', labelsize = med)   # fontsize of the x & y labels
plt.rc('xtick', labelsize = sm)   # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)   # fontsize of the tick labels
plt.rc('legend', fontsize = sm)   # legend fontsize
plt.rc('figure', titlesize = lg)  # fontsize of the figure title
plt.rc('axes', linewidth=2)       # linewidth of plot lines
plt.rcParams['figure.figsize'] = [13, 5]
plt.rcParams['figure.dpi'] = 150

class MovingAverage1(object):
    def __init__(self, period='1y', interval='1d'):
        self.period, self.interval = period, interval

    def getData(self):
        self.loaded_lst = read_pickle(self.path)
        print(self.loaded_lst)
        return self.loaded_lst

    def mAvg(self):
        original_df = self.getData()
        self.df = pd.DataFrame(original_df['Close'])
        self.df_daily_pct_c = self.df.pct_change()
        self.df_daily_pct_c.fillna(0, inplace = True)
        self.df_daily_pct_c = self.df / self.df.shift(1) - 1
        self.df['Daily_S_RoR'] = self.df_daily_pct_c * 100  
            # LOG Rate Of Return
        self.df_daily_log_returns = np.log(self.df.pct_change() + 1)
        self.df_daily_log_returns.fillna(0, inplace = True)
        self.df['Daily_Log'] = self.df_daily_log_returns['Close'] * 100
            # Total Return
        self.df_cum_daily_return = (1 + self.df_daily_pct_c).cumprod() 
        self.df['Total_RoR'] = self.df_cum_daily_return
        self.df.rename(columns={'Close': self.ticker}, inplace=True)
            # Build MovingAverages
        self.short_window, self.long_window, self.period, self.multiplier = 5, 22, 20, 2
        self.signals = pd.DataFrame(index=self.df.index)
        self.signals['signal'] = 0.0
        self.signals[self.ticker] = self.df[self.ticker]
        self.signals['short_mavg'] = self.df[self.ticker].rolling(window=self.short_window,min_periods=1,center=False).mean()
        self.signals['long_mavg'] = self.df[self.ticker].rolling(window=self.long_window, min_periods=1, center=False).mean()
        self.signals['signal'][self.short_window:] = np.where(self.signals['short_mavg'][self.short_window:] > self.signals['long_mavg'][self.short_window:],1.0, 0.0)
        self.signals['positions'] = self.signals['signal'].diff()
        self.signals['UpperBand'] = self.df[self.ticker].rolling(self.period).mean() + self.df[self.ticker].rolling(self.period).std() * self.multiplier
        self.signals['LowerBand'] = self.df[self.ticker].rolling(self.period).mean() - self.df[self.ticker].rolling(self.period).std() * self.multiplier

    def plot_mAvg(self, ticker, path):
        self.ticker = ticker
        self.path = path
        self.mAvg()
        fig, ax1 = plt.subplots()
        self.df[self.ticker].plot(ax=ax1, lw=2, color = 'k')
        ax1.plot(self.signals.short_mavg, '--', lw=1.5, color = 'blue')
        ax1.plot(self.signals.long_mavg, '--', lw=1.5, color = 'green')
        ax1.plot(self.signals[['UpperBand','LowerBand']], '-', color = 'red', lw=1.5)
        ax1.plot(self.signals.loc[self.signals.positions == 1.0].index,self.signals.short_mavg[self.signals.positions == 1.0],'^', markersize=11, color = 'g')
        ax1.plot(self.signals.loc[self.signals.positions == -1.0].index, self.signals.short_mavg[self.signals.positions == -1.0],'v', markersize=11, color = 'r')
        ax1.vlines(self.signals.short_mavg[self.signals.positions == 1.0].index, self.df[self.ticker].min(), self.df[self.ticker].max(), linestyles ="solid", colors ="green", lw=1.5)
        ax1.vlines(self.signals.short_mavg[self.signals.positions == -1.0].index, self.df[self.ticker].min(), self.df[self.ticker].max(), linestyles ="solid", colors ="purple", lw=1.5)
        ax1.set_ylabel(ylabel='Price in $')
        ax1.set_title(self.ticker+' - Moving Average Trade Signals (2-20)')
        ax1.legend(self.signals[[self.ticker, 'short_mavg','long_mavg','UpperBand','LowerBand']])
        plt.tight_layout()
        plt.show()
        print('Time To Buy:\n  ', self.signals.short_mavg[self.signals.positions == 1.0], '\n\nTime To Sell: \n', self.signals.short_mavg[self.signals.positions == -1.0],'\n')

    def gainers(self):
        print(get_day_gainers().Symbol[:5])
        # for g in get_day_losers().Symbol[:5]:
        #     self.signal = self.plot_mAvg(ticker = g)

    def losers(self):
        print(get_day_losers().Symbol[:5])
        # for l in get_day_losers().Symbol[:5]:
        #     self.signal = self.plot_mAvg(ticker = l)

if __name__ == "__main__":
    mavg = MovingAverage1()
    ticker = '^GSPC'
    path = '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/raw/^GSPC_data_10y_1d.pkl'
    mavg.plot_mAvg(ticker, path)
    mavg.gainers()
    mavg.losers()