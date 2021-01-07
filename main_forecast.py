from src.data.make_dataset import Get_Historical_Data
from models.arima import Arima_Model
from models.movingAverage1 import MovingAverage1
from models.movingAverage2 import MovingAverage2
from models.movingAverage3 import MovingAverage3
from models.efficientFrontier import Efficient_Frontier
from models.lstm_rnn import LSTM_RNN
from models import optimizer
from models.optimizer import Look_At_Optimized_Portfolios
from models import optimizer2
from models.sckitLearn_stockAnalysis import SckitLearn_Stock_Analysis
from models.strategy import Strategy1
from models.strategy3 import Strategy3
from models.univariate_TS_regression import Univariate_TS_Reg
from models.multivariate_TS import Multivariate_TS
from models.viz_model_predict_SARIMAX import Model
import pandas as pd
import numpy as np
from pandas.io.pickle import read_pickle
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_palette('cubehelix')
# sns.set(style='darkgrid', context='talk', palette='Dark2')
# plt.style.use('seaborn-colorblind')
plt.style.use('seaborn') 
sns.set_style('whitegrid')
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
plt.rcParams['figure.dpi'] = 125
from pathlib import Path
path = Path.cwd()
from yahoo_fin.stock_info import tickers_dow, tickers_nasdaq, tickers_other, tickers_sp500

if __name__ == '__main__':
    pass
# PULL PRICES & TICKERS
    # p = str(path) + '/data/'
    # tic = '^GSPC'
    # tics = ['AAPL','AMZN','MSFT','TSLA','NFLX','^GSPC']
    # run = Get_Historical_Data(path=p, period='10y', interval='1d')
    # run.getData_S(tic)
    # names = ['dow','other']
    # run.getData_M(tics) #, save_name='portfolio_data_')
    # run.getData_I(names)
    # run.getData_I_tics()

# LSTM_RNN:
    path = '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/raw/'
    ticker = '^GSPC'
    period='10y'
    interval='1d'
    lstm_rnn = LSTM_RNN(ticker, path, period, interval)
    lstm_rnn.viz(epochs = 5, batch_size = 13, loss = 'mean_squared_error')

# ARIMA MODEL:
    ticker = '^GSPC'
    run = Arima_Model(ticker)
    run.model_arima()

# VIZ_MODEL_PREDICTT_SP500-INDEX (PCA, SEASONAL-DECOMPOSITION, SARIMAX):
    p = '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/'
    stock=read_pickle(p + 'tickers/sp500_ticker_list.pkl')
    x = Model(stock)
    x.predict()

# UNIVARIATE TIME SERIES - REGRESSION:
    ticker = '^GSPC'
    run = Univariate_TS_Reg(ticker)
    run.runs()

# MULTIVARIATE TIME SERIES:
    run = Multivariate_TS()
    run.multivariate()

# SCKITLEARN-ANALYSIS & PREDICT (Lin Regression, Quadratic Regression 2&3, KNN Regression)
    ticker = '^GSPC'
    run = SckitLearn_Stock_Analysis(ticker)
    run.model_out()    