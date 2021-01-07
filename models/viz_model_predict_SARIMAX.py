import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.pickle import read_pickle
plt.style.use('ggplot')
sm, med, lg = 10, 15, 20
plt.rc('font', size = sm)         # controls default text sizes
plt.rc('axes', titlesize = med)   # fontsize of the axes title
plt.rc('axes', labelsize = med)   # fontsize of the x & y labels
plt.rc('xtick', labelsize = sm)   # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)   # fontsize of the tick labels
plt.rc('legend', fontsize = sm)   # legend fontsize
plt.rc('figure', titlesize = lg)  # fontsize of the figure title
plt.rc('axes', linewidth=2)       # linewidth of plot lines
import yfinance as yf
from yahoo_fin.stock_info import * 
from sklearn.decomposition import KernelPCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools    
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")
from scipy import stats
from datetime import *
today = date.today()
p = '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/'
print('\n          * * * NO ISSUES - ALL IMPORTS LOADED * * * \n')

class Model:
    def __init__(self, stock):
        self.stock = stock

    def dataHull(self):
        # self.df_components = yf.download(
        #     self.stock, 
        #     start='2019-01-03', 
        #     end='2019-12-30', 
        #     interval='1d',
        #     parse_dates=True, 
        #     index_col=0
        #     )['Adj Close']
        # self.df_components.to_pickle(p + 'raw/sp500component_hist_2019.pkl')
        self.dfc = pd.read_pickle(p + 'raw/sp500component_hist_2019.pkl')
        self.dfc = self.dfc[:-1]
        self.dfc.index = pd.to_datetime(self.dfc.index)

        self.filled_df_components = self.dfc.fillna(method='ffill')
        self.daily_df_components = self.filled_df_components.resample('24h').ffill()
        self.daily_df_components = self.daily_df_components.fillna(method='bfill')

        self.spData = pd.read_pickle(p + 'raw/^GSPC_data_10y_1d.pkl')
        self.dataSP = pd.DataFrame(self.spData['Close'])
        self.dataSP.columns = ['SP500']
        self.dataSP.index = pd.to_datetime(self.dataSP.index)
        self.sp_2017 = pd.DataFrame(self.dataSP.loc['2019-01-03':'2019-12-27'])
        self.sp_2017 = self.sp_2017.resample('24h').ffill()

    def kernel_pca(self):
        self.dataHull()
        self.fn_z_score = lambda x: (x - x.mean()) / x.std()
        self.df_z_components = self.daily_df_components.apply(self.fn_z_score)
        self.df_z_components = -self.df_z_components.fillna(0)
        self.fitted_pca = KernelPCA().fit(self.df_z_components)
        plt.rcParams['figure.figsize'] = (12,8)
        plt.plot(self.fitted_pca.lambdas_)
        plt.ylabel('eigenvalues')
        plt.show()

        self.fn_weighted_avg = lambda x: x / x.sum()
        self.weighted_values = self.fn_weighted_avg(self.fitted_pca.lambdas_)[:13]
        print(self.weighted_values)
        print(self.weighted_values.sum())

        self.daily_df_components = self.daily_df_components.fillna(0)
        self.kernel_pca = KernelPCA(n_components = 13).fit(self.df_z_components)
        self.pca_13 = self.kernel_pca.transform(self.daily_df_components)
        self.weights = self.fn_weighted_avg(self.kernel_pca.lambdas_)
        self.reconstructed_values = list(np.dot(self.pca_13, self.weights))

        self.fn_z_score = lambda x: (x - x.mean()) / x.std()
        self.df_combined = self.sp_2017.copy()
        self.df_combined['pca_13'] = self.reconstructed_values
        self.df_combined = self.df_combined.apply(self.fn_z_score)
        self.df_combined.plot(figsize=(12, 8))
        plt.legend()
        plt.show()

        self.df_settle = self.spData['Close'].resample('MS').ffill().dropna()
        self.df_rolling = self.df_settle.rolling(12)
        self.df_mean = self.df_rolling.mean()
        self.df_std = self.df_rolling.std()
        # plt.figure(figsize=(12, 8))
        # plt.plot(self.df_settle, label='Original')
        # plt.plot(self.df_mean, label='Mean')
        # plt.legend()
        # plt.show()

        # self.df_std.plot(figsize=(12, 8))
        # plt.show()

    def adf(self):
        self.kernel_pca()
        self.result = adfuller(self.df_settle)
        print('ADF statistic: ',  self.result[0])
        print('p-value:', self.result[1])
        self.critical_values = self.result[4]
        for key, value in self.critical_values.items():
            print('Critical value (%s): %.3f' % (key, value))
        self.df_log = np.log(self.df_settle)
        self.df_log_ma= self.df_log.rolling(2).mean()
        self.df_detrend = self.df_log - self.df_log_ma
        self.df_detrend.dropna(inplace=True)
            # Mean and standard deviation of detrended data
        self.df_detrend_rolling = self.df_detrend.rolling(12)
        self.df_detrend_ma = self.df_detrend_rolling.mean()
        self.df_detrend_std = self.df_detrend_rolling.std()
            # Plot
        # plt.figure(figsize=(12, 8))
        # plt.plot(self.df_detrend, label='Detrended')
        # plt.plot(self.df_detrend_ma, label='mean')
        # plt.plot(self.df_detrend_std, label='std')
        # plt.legend(loc='upper right')
        # plt.show()
        self.result2 = adfuller(self.df_detrend)
        print('ADF statistic: ', self.result2[0])
        print('p-value: %.5f' % self.result2[1])
        self.critical_values2 = self.result2[4]
        for key, value in self.critical_values2.items():
            print('Critical value (%s): %.3f' % (key, value))
        self.df_log_diff = self.df_log.diff(periods=3).dropna()
            # Mean and standard deviation of differenced data
        self.df_diff_rolling = self.df_log_diff.rolling(12)
        self.df_diff_ma = self.df_diff_rolling.mean()
        self.df_diff_std = self.df_diff_rolling.std()
            # Plot the stationary data
        # plt.figure(figsize=(12, 8))
        # plt.plot(self.df_log_diff, label='Differenced')
        # plt.plot(self.df_diff_ma, label='mean')
        # plt.plot(self.df_diff_std, label='std')
        # plt.legend(loc='upper right')
        # plt.show()

    def seasonal_decomp(self):
        self.adf()
        self.decompose_result = seasonal_decompose(self.df_log.dropna(), period=12)
        self.df_trend = self.decompose_result.trend
        self.df_season = self.decompose_result.seasonal
        self.df_residual = self.decompose_result.resid
        plt.rcParams["figure.figsize"] = (12, 8)
        fig = self.decompose_result.plot()
        plt.show()       
        self.df_log_diff = self.df_residual.diff().dropna()
        # Mean and standard deviation of differenced data
        self.df_diff_rolling = self.df_log_diff.rolling(12)
        self.df_diff_ma = self.df_diff_rolling.mean()
        self.df_diff_std = self.df_diff_rolling.std()
        # Plot the stationary data
        # plt.figure(figsize=(12, 8))
        # plt.plot(self.df_log_diff, label='Differenced')
        # plt.plot(self.df_diff_ma, label='Mean')
        # plt.plot(self.df_diff_std, label='Std')
        # plt.legend()
        # plt.show()
        self.result = adfuller(self.df_residual.dropna())
        print('ADF statistic:',  self.result[0])
        print('p-value: %.5f' % self.result[1])
        self.critical_values = self.result[4]
        for key, value in self.critical_values.items():
            print('Critical value (%s): %.3f' % (key, value))

    def arima_grid_search(self, s=12):
        self.seasonal_decomp()
        self.s = s       
        self.p = self.d = self.q = range(2)
        self.param_combinations = list(itertools.product(self.p, self.d, self.q))
        self.lowest_aic, self.pdq, self.pdqs = None, None, None
        self.total_iterations = 0
        for order in self.param_combinations:    
            for (self.p, self.q, self.d) in self.param_combinations:
                self.seasonal_order = (self.p, self.q, self.d, self.s)
                self.total_iterations += 1
                try:
                    self.model = SARIMAX(self.df_settle, order=order, 
                        seasonal_order=self.seasonal_order, 
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        disp=False)
                    self.model_result = self.model.fit(maxiter=200, disp=False)
                    if not self.lowest_aic or self.model_result.aic < self.lowest_aic:
                        self.lowest_aic = self.model_result.aic
                        self.pdq, self.pdqs = order, self.seasonal_order
                except Exception as ex:
                    continue
        print('ARIMA{}x{}'.format(self.pdq, self.pdqs))
        print('Lowest AIC: ' , self.lowest_aic)
        return self.lowest_aic, self.pdq, self.pdqs

    def fitModel_to_SARIMAX(self):
        self.arima_grid_search()
        self.model = SARIMAX(
            self.df_settle,
            order=self.pdq,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            disp=False)
        self.model_results = self.model.fit(maxiter = 200, disp = False)
        print(self.model_results.summary())
        self.model_results.plot_diagnostics(figsize=(13, 13))
        plt.show()
        return self.model_results

    def predict(self):
        self.fitModel_to_SARIMAX()
        self.n = len(self.df_settle.index)
        self.prediction = self.model_results.get_prediction(
            start = self.n - 12*5,
            end = self.n + 6)
        self.prediction_ci = self.prediction.conf_int()
        plt.figure(figsize=(12, 6))
        ax = self.df_settle['2010':].plot(label='actual')
        self.prediction_ci.plot(
            ax=ax, style=['--', '--'],
            label='predicted/forecasted')
        self.ci_index = self.prediction_ci.index
        self.lower_ci = self.prediction_ci.iloc[:, 0]
        self.upper_ci = self.prediction_ci.iloc[:, 1]
        ax.fill_between(self.ci_index, self.lower_ci, self.upper_ci,
            color='r', alpha=.1)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Prices')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    pass
    # stock=read_pickle(p + 'tickers/sp500_ticker_list.pkl')
    # x = Model(stock)
    # x.predict()