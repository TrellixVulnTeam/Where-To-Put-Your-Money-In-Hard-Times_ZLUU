from pandas.io.pickle import read_pickle
from src.data.make_dataset import Get_Historical_Data
from models.movingAverage1 import MovingAverage1
from models.movingAverage2 import MovingAverage2
from models.movingAverage3 import MovingAverage3
from models.efficientFrontier import Efficient_Frontier
from models.lstm_rnn import LSTM_RNN
from models import optimizer
from models.optimizer import Look_At_Optimized_Portfolios
from models import optimizer2
from models.viz_model_predict_SARIMAX import Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('cubehelix')
# sns.set(style='darkgrid', context='talk', palette='Dark2')
plt.style.use('seaborn') # plt.style.use('seaborn-colorblind') #alternative
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
    
# movingAverage.py ~ MOVING_AVERAGES:
    # mavg = MovingAverage1()
    # p = str(path) + '/data/raw/'
    # ticker = '^GSPC'
    # mavg.plot_mAvg(ticker, p)
    # mavg.gainers()
    # mavg.losers()

    # x = MovingAverage2()
    # name = 'AAPL'
    # period = '3mo's
    # x.setup(name, period)
    # x.level()

    # x = MovingAverage3()
    # stock_df, table = x.MovingAverageCrossStrategy('AAPL', 20, 50, 'SMA', '1y')
    # stock_df, table = x.MovingAverageCrossStrategy('AAPL', 20, 50, 'EMA', '1y') 
    
# EFFICIENT-FRONTIER:
    # N_PORTFOLIOS = 10 ** 5
    # N_DAYS = 252
    # RISKY_ASSETS = ['AAPL', 'TSLA', 'BLDP', 'BE', 'SAGE']
    # RISKY_ASSETS.sort()
    # START_DATE = '2019-01-01'
    # END_DATE = '2020-12-24'
    # n_assets = len(RISKY_ASSETS)
    # ef = Efficient_Frontier(N_PORTFOLIOS, N_DAYS, RISKY_ASSETS, START_DATE, END_DATE, n_assets)
    # ef.final_plot()

# LSTM_RNN:
    # path = '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/raw/'
    # ticker = '^GSPC'
    # period='10y'
    # interval='1d'
    # lstm_rnn = LSTM_RNN(ticker, path, period, interval)
    # lstm_rnn.viz(epochs = 5, batch_size = 13, loss = 'mean_squared_error')  

# VIZ_MODEL_PREDICTT_SP500-INDEX:
    # p = '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/'
    # stock=read_pickle(p + 'tickers/sp500_ticker_list.pkl')
    # x = Model(stock)
    # x.predict()

# PORTFOLIO OPTIMIZERS:
    # saveName = 'sample_data' 'sp500' 'dow' 'nasdaq' 'sample' 'broker_pos_data' 'roth_pos_data', 'moveOn_pos_data', 'potential_pos_data'
    # saveName = 'sp500'
    # p = "/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/"
    # path = p + 'raw/' + saveName + '_10y_1d.pkl'
    # PT_data = pd.read_pickle(path)
    # PT = pd.DataFrame(PT_data)
    # tickers, returns = list(PT.columns), PT.pct_change()
    # mean_returns, cov_matrix, num_portfolios, risk_free_rate = returns.mean(), returns.cov(), 25000, 0.0178
    # destination = "/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/processed/" 

    # rp, sdp, rp_min, sdp_min = optimizer.display_ef_with_selected(PT, mean_returns, cov_matrix, risk_free_rate, destination, saveName, returns, num_portfolios) #####
    # x = Look_At_Optimized_Portfolios(saveName) #####
    # df, fd, a, b = x.viz()
    # print(f'\nMaximum Sharpe Ratio Portfolio: \n   Annualized Return = {round(rp,2)}%\n   Annualized Volatility = {round(sdp,2)}%\n\n {df.iloc[a]}')
    # print(f'\n\nMinimum Volatility Portfolio \n   Annualized Return = {round(rp_min,2)}%\n   Annualized Volatility = {round(sdp_min,2)}%\n\n {fd.iloc[b]}')
 
    # optimizer2.display_ef_with_selected(PT, mean_returns, cov_matrix, risk_free_rate, destination, saveName, returns, num_portfolios) #####

# STRATEGIES:
    # from models.strategy import Strategy1
    # from models.strategy3 import Strategy3
    # Strategy1()
    # Strategy3() 