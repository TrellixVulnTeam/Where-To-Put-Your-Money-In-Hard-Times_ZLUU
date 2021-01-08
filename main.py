import warnings
warnings.filterwarnings('ignore')
from data.make_dataset import Get_Historical_Data
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
from models import prophet

from fbprophet import Prophet
m = Prophet(daily_seasonality = True)
import pandas as pd
import numpy as np
from pandas.io.pickle import read_pickle
import matplotlib.pyplot as plt
plt.style.use('seaborn') 
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
plt.rcParams['figure.dpi'] = 134
from pathlib import Path
path = Path.cwd()

if __name__ == '__main__': # * * * * * * * # ~ STANDARD INPUTS ~ # * * * * * * * #
    single_ticker = 'AAPL'
    mini_portfolio = ['AAPL','AMZN','MSFT','TSLA','NFLX']
    index_tickers = ['^GSPC', '^DJI', '^IXIC','^NYA','^RUT']
    sp500index = '^GSPC'
    sp500_trackers = ['^GSPC','SPLG','IVV','VOO','SPY','FUSEX','SWPPX','VFINX']
    dow_trackers = ['^DJI','DIA']
    nasdaq_trackers = ['^IXIC','QQQ','IEF','PGJ']
    nyse_trackers = ['^NYA','VGT','GLD','VTI']
    russell_trackers = ['^RUT','IWM','VTWO','TNA','UWM','TZA','URTY','RWM','SRTY','TWM']
    PATH = str(path) + '/data/'
    start='2010-01-01'
    end='2018-01-01'

if __name__ == '__main__': 
    pass# * * * * * * * # ### ~ PULL PRICES & TICKERS ~ ### # * * * * * * * #
    # run_get_hist_data = Get_Historical_Data(path=PATH, period='10y', interval='1d')
    # get_hist_data_names = ['dow','sp500','nasdaq','other']
    # run_get_hist_data.getData_S(sp500index)
    # run_get_hist_data.getData_M(index_tickers)
    # run_get_hist_data.getData_I(get_hist_data_names)
    # # run_get_hist_data.getData_I_tics()

if __name__ == '__main__': 
    pass # * * * * * * * # ### ~ MOVING_AVERAGES ~ # * * * * * * * ## * * * * * * * #
    # mavg = MovingAverage1(sp500index, PATH, period='1y').gainers_losers_crypto()
    # mavg2 = MovingAverage2(sp500index, period='6mo').level()
    # stock_df_sma, table_sma = MovingAverage3(sp500index, short_window=2, long_window=20, moving_avg='SMA', period='1y').MovingAverageCrossStrategy()
    # stock_df_ema, table_ema = MovingAverage3(sp500index, short_window=2, long_window=20, moving_avg='EMA', period='1y').MovingAverageCrossStrategy()
    # plt.show()

if __name__=='__main__':
    pass ### ### ### ~ STRATEGIES ~ ### ### ###
    # Strategy1(sp500index, PATH).model()
    # Strategy3(sp500index, 'SP500', PATH).model3()

if __name__ == '__main__':
    pass# * * * * * * * ## * * * * * * * # #~ PORTFOLIO OPTIMIZERS: ~ # * * * * * * * ## * * * * * * * #
    # saveName = 'sample_data' 'sp500' 'dow' 'nasdaq' 'sample' 'broker_pos_data' 'roth_pos_data', 'moveOn_pos_data', 'potential_pos_data'
    # saveName = 'dow'
    # path = PATH + 'raw/' + saveName + '_10y_1d.pkl'
    # PT_data = pd.read_pickle(path)
    # PT = pd.DataFrame(PT_data)
    # tickers, returns = list(PT.columns), PT.pct_change()
    # mean_returns, cov_matrix, num_portfolios, risk_free_rate = returns.mean(), returns.cov(), 25000, 0.0104
    # destination = PATH + '/processed/'

    # rp, sdp, rp_min, sdp_min = optimizer.display_ef_with_selected(PT, mean_returns, cov_matrix, risk_free_rate, dest, saveName, returns, num_portfolios)
    # df, fd, a, b = Look_At_Optimized_Portfolios(saveName).viz()
    # print(f'\nMaximum Sharpe Ratio Portfolio: \n   Annualized Return = {round(rp,2)}%\n   Annualized Volatility = {round(sdp,2)}%\n\n {df.iloc[a]}')
    # print(f'\n\nMinimum Volatility Portfolio \n   Annualized Return = {round(rp_min,2)}%\n   Annualized Volatility = {round(sdp_min,2)}%\n\n {fd.iloc[b]}')
 
    # optimizer2.display_ef_with_selected(PT, mean_returns, cov_matrix, risk_free_rate, destination, saveName, returns, num_portfolios)    

if __name__ == '__main__':
    pass# *## ~ EFFICIENT-FRONTIER: ~# * * * * * * * ## * * * * * * * #
    # N_PORTFOLIOS = 10 ** 5
    # N_DAYS = 252
    # RISKY_ASSETS = ['^GSPC', 'TSLA', 'AAPL', 'AMZN', 'NFLX']
    # RISKY_ASSETS.sort()
    # START_DATE = '2019-01-01'
    # END_DATE = '2020-12-24'
    # n_assets = len(RISKY_ASSETS)
    # ef = Efficient_Frontier(N_PORTFOLIOS, N_DAYS, RISKY_ASSETS, START_DATE, END_DATE, n_assets)
    # ef.final_plot()

if __name__ == '__main__':
    pass# * * * * * * * ## * * * * * * * #### ~ LSTM_RNN: ~# * * * * * * * ## * * * * * * * #
    # LSTM_RNN(sp500index, PATH+'raw/').viz(epochs = 5, batch_size = 13, loss = 'mean_squared_error')

if __name__ == '__main__':
    pass# * * * * * * * ## * * * * * * * #### ~ ARIMA MODEL: ~
    # Arima_Model(sp500index).model_arima()

if __name__ == '__main__':
    pass# * * * * * * * ## * * * * * * * #### ~ VIZ_MODEL_PREDICTT_SP500-INDEX (PCA, SEASONAL-DECOMPOSITION, SARIMAX): ~
    # stock=read_pickle(PATH + 'interim/tickers/sp500_ticker_list.pkl')
    # Model(stock).predict()

if __name__ == '__main__':
    pass# * * * * * * * ## * * * * * * * #### ~ UNIVARIATE TIME SERIES - REGRESSION: ~
    # Univariate_TS_Reg(sp500index).runs()


if __name__ == '__main__':
    pass# * * * * * * * ## * * * * * * * #### ~ MULTIVARIATE TIME SERIES: ~
    # Multivariate_TS().multivariate()


if __name__ == '__main__':
    pass# ~ SCKITLEARN-ANALYSIS & PREDICT (Lin Regression, Quadratic Regression 2&3, KNN Regression) ~
    # SckitLearn_Stock_Analysis(sp500index).model_out()    


if __name__ == '__main__':
    pass# * * * * * * * ## * * * * * * * #### ~ PROPHET MODEL: ~ 
    # prophet.Prophet(sp500index, m, start, end).model_prophet()