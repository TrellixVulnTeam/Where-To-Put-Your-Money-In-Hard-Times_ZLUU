from src.data.make_dataset import Get_Historical_Data
from src.models.movingAverage1 import MovingAverage1
# from src.models.movingAverage2 import MovingAverage2
# from src.models.movingAverage3 import MovingAverage3
# from src.models.movingAverage4 import MovingAverageCrossStrategy
# from src.models.efficientFrontier import Efficient_Frontier
# from src.models.lstm_rnn import LSTM_RNN
# from src.models import optimizer
# from src.models.sma_vectorized_backtest import SMAVectorBacktester
# from src.models.viz_model_predict_SARIMAX import Model

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('cubehelix')
sns.set(style='darkgrid', context='talk', palette='Dark2')
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
    # mavg.plot_mAvg('^GSPC', '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/raw/^GSPC_data_10y_1d.pkl')
    # mavg.plot_mAvg(tic)

    # x = MovingAverage2()
    # period = '3mo'
    # x.setup(tic, period)
    # x.level()

    # y = MovingAverage3()
    # stock_df, table = y.MovingAverageCrossStrategy(tic, 2, 20)
    # print(table)  

    # z = MovingAverageCrossStrategy(
    #     stock_symbol = 'TSLA', 
    #     start_date = '2020-01-01', 
    #     short_window = 20, 
    #     long_window = 50, 
    #     moving_avg = 'SMA', 
    #     display_table = True   
    # )    
    
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
    # lstm_rnn = LSTM_RNN('TSLA', period='5y', interval='1d')
    # lstm_rnn.viz(epochs = 50, batch_size = 13, loss = 'mean_squared_error')  

# VIZ_MODEL_PREDICTT_SP500-INDEX:
    # p = '/home/gordon/work/assemble/data/raw/'
    # df = pd.read_csv(p + 'stock_history/spComponentData.csv', index_col='Date')
    # stock = list(df.columns)
    # x = Model(stock)
    # x.dataHull()
    # x.Kernel_pca()
    # x.adf()
    # x.seasonal_decomp()
    # lowest_aic, order, seasonal_order = x.arima_grid_search(12)
    # print('ARIMA{}x{}'.format(order, seasonal_order))
    # print('Lowest AIC: ' , lowest_aic)
    # mod_res = x.fitModel_to_SARIMAX()
    # print(mod_res.summary()) 
    # mod_res.plot_diagnostics(figsize=(12, 8))
    # plt.show()
    # x.predict()    