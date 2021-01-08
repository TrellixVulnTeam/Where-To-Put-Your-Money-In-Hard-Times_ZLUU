import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt


class MovingAverage2(object):
    def __init__(self, name, period='1y'):
        self.name, self.period = name, period
        ticker = yfinance.Ticker(self.name)
        self.df = ticker.history(period=self.period)
        self.df['Date'] = pd.to_datetime(self.df.index)
        self.df['Date'] = self.df['Date'].apply(mpl_dates.date2num)
        self.df = self.df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]


    def isSupport(self,df,i):
        support = self.df['Low'][i] < self.df['Low'][i-1]  and self.df['Low'][i] < self.df['Low'][i+1] \
        and self.df['Low'][i+1] < self.df['Low'][i+2] and self.df['Low'][i-1] < self.df['Low'][i-2]
        return support


    def isResistance(self,df,i):
        resistance = self.df['High'][i] > self.df['High'][i-1] and self.df['High'][i] > self.df['High'][i+1] \
        and self.df['High'][i+1] > self.df['High'][i+2] and self.df['High'][i-1] > self.df['High'][i-2] 
        return resistance


    def plot_all(self):
        fig, ax = plt.subplots()
        candlestick_ohlc(ax,self.df.values,width=0.6, colorup='green', colordown='red', alpha=0.8)
        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        # fig.tight_layout()
        for level in self.levels:
            plt.hlines(level[1],xmin=self.df['Date'][level[0]], xmax=max(self.df['Date']),colors='blue')
            plt.title(self.name + ' Support & Resistance Price Levels')
            plt.tight_layout()
            plt.grid(True, linestyle='--')
        fig.show()
        

    def isFarFromLevel(self,l):
        return np.sum([abs(l-x) < self.s  for x in self.levels]) == 0


    def level(self):
        self.levels = []
        for i in range(2, self.df.shape[0]-2):
            if self.isSupport(self.df,i):
                self.levels.append((i, self.df['Low'][i]))
            elif self.isResistance(self.df, i):
                self.levels.append((i,self.df['High'][i]))
        self.s =  np.mean(self.df['High'] - self.df['Low'])
        self.levels = []
        for i in range(2, self.df.shape[0]-2):
            if self.isSupport(self.df, i):
                l = self.df['Low'][i]
                if self.isFarFromLevel(l):
                    self.levels.append((i,l))
            elif self.isResistance(self.df,i):
                l = self.df['High'][i]
                if self.isFarFromLevel(l):
                    self.levels.append((i,l))
        print(self.levels)
        self.plot_all()
        plt.show()


if __name__ == '__main__':
    pass
    # x = MovingAverage2()
    # name = '^GSPC'
    # name = '^NYA'
    # period = '3mo'
    # x.setup(name, period)
    # x.level()