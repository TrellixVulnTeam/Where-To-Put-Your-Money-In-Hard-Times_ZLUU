from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime
import math
import pandas_datareader.data as web
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
mpl.rc('figure', figsize=(8, 7))
mpl.__version__
style.use('ggplot')


class SckitLearn_Stock_Analysis(object):
    def __init__(self, ticker):
        self.ticker = ticker

    def model_out(self):
        start = datetime.datetime(2010, 1, 1)
        end = datetime.datetime(2018, 1, 1)
        df = yf.download(self.ticker, start=start, end=end, parse_dates=True)

        sp = yf.download('^GSPC',start='2010-01-01', end='2019-03-01')['Adj Close']
        sp.columns = ['sp_actual']
        
        close_px = df['Adj Close']
        mavg = close_px.rolling(window=100).mean()
        close_px.plot(label='^GSPC')
        mavg.plot(label='mavg')
        plt.legend()
        plt.title(f'Moving Average vs {self.ticker} Price')
        plt.show()

        rets = close_px / close_px.shift(1) - 1
        # rets.plot(label='return')
        # plt.title(f'Returns of {self.ticker} vs Time')
        # plt.show()
        dfcomp = yf.download(
            ['^GSPC', 'AAPL', 'DIS', 'MSFT'], start=start, end=end)['Adj Close']
        retscomp = dfcomp.pct_change()
        corr = retscomp.corr() 
        # plt.scatter(
        #     retscomp.AAPL, retscomp['^GSPC'])
        # plt.title(f'ScatterPlot Featuring Returns of {self.ticker} vs AAPL')
        # plt.xlabel('Returns AAPL')
        # plt.ylabel('Returns SP500')
        # plt.show()

        # scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10))
        # plt.title(f'ScatterPlot Matrix vs {self.ticker}')
        # plt.show()

        # plt.imshow(corr, cmap='hot', interpolation='none')
        # plt.colorbar()
        # plt.xticks(range(len(corr)), corr.columns)
        # plt.yticks(range(len(corr)), corr.columns)
        # plt.title(f'HeatMap - Correlation of Returns To: {self.ticker}')
        # plt.show()

        plt.scatter(retscomp.mean(), retscomp.std())
        plt.xlabel('Expected returns')
        plt.ylabel('Risk')
        for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
            plt.annotate(
                label, 
                xy = (x, y), xytext = (20, -20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.title(f'Expected Returns Plot Verses {self.ticker}')
        plt.show()

        dfreg = df.loc[:,['Adj Close','Volume']]
        dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
        dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

            # Drop missing value
        dfreg.fillna(value=-99999, inplace=True)
            # We want to separate 1 percent of the data to forecast
        forecast_out = int(math.ceil(0.01 * len(dfreg)))
            # Separating the label here, we want to predict the AdjClose
        forecast_col = 'Adj Close'
        dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
        X = np.array(dfreg.drop(['label'], 1))
            # Scale the X so that everyone can have the same distribution for linear regression
        X = sklearn.preprocessing.scale(X)
            # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
            # Separate label and identify it as y
        y = np.array(dfreg['label'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Linear regression
        clfreg = LinearRegression(n_jobs=-1)
        clfreg.fit(X_train, y_train)

        # Quadratic Regression 2
        clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
        clfpoly2.fit(X_train, y_train)

        # Quadratic Regression 3
        clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
        clfpoly3.fit(X_train, y_train)

        # KNN Regression
        clfknn = KNeighborsRegressor(n_neighbors=2)
        clfknn.fit(X_train, y_train)
        confidencereg = clfreg.score(X_test, y_test)
        confidencepoly2 = clfpoly2.score(X_test,y_test)
        confidencepoly3 = clfpoly3.score(X_test,y_test)
        confidenceknn = clfknn.score(X_test, y_test)
        # results
        print('\nThe linear regression confidence is ', confidencereg),
        print('The quadratic regression 2 confidence is ', confidencepoly2),
        print('The quadratic regression 3 confidence is ', confidencepoly3),
        print('The knn regression confidence is ', confidenceknn, '\n')

        forecast_set = clfreg.predict(X_lately)
        dfreg['Forecast'] = np.nan
        dfreg['Forecast']
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        days = 1
        next_unix = last_unix + datetime.timedelta(days)
        for i in forecast_set:
            next_date = next_unix
            next_unix += datetime.timedelta(days=8)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
        
        fig = plt.subplots(figsize=(15,6), dpi=150)
        plt.plot(dfreg['Adj Close'].tail(500), lw=2, ls='-', color='b')
        plt.plot(dfreg['Forecast'].tail(500), lw=2, ls='-', color='g')
        plt.plot(sp,label='sp500-actual', color='r', ls='--', lw=1)
        plt.legend(loc=4)
        plt.title('6 MONTH FORECAST OUT OF SP500 INDEX')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

if __name__ == '__main__':
    ticker = '^GSPC'
    run = SckitLearn_Stock_Analysis(ticker)
    run.model_out()