from sklearn.linear_model import LinearRegression  
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import datetime as dt
from pylab import mpl, plt
import yfinance as yf
from itertools import product
plt.style.use('seaborn')
# mpl.rcParams['font.family'] = 'serif'

class Strategy3:
    symbol = 'SP500'
    tic = '^GSPC'
    ticker = yf.Ticker(tic)
    raw = ticker.history(period='10y', interval='1d')
    raw.columns = ['Open', 'High', 'Low', symbol, 'Volume', 'Dividends', 'Stock Splits']


    SMA1 = 20
    SMA2 = 50
    data1= pd.DataFrame(raw[symbol])
    data1.columns = [symbol]
    data1['SMA1'] = data1[symbol].rolling(SMA1).mean()
    data1['SMA2'] = data1[symbol].rolling(SMA2).mean()
    data1.plot(figsize=(10,5))
    plt.show()

    data1['Position'] = np.where(data1['SMA1'] > data1['SMA2'], 1, -1)
    ax = data1.plot(secondary_y='Position', figsize=(10,6))
    ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
    plt.show()

    data1['Returns'] = np.log(data1[symbol] / data1[symbol].shift(1))
    data1['Strategy'] = data1['Position'].shift(1) * data1['Returns']
    data1.round(4).tail()
    data1.dropna(inplace=True)
    print('')
    np.exp(data1[['Returns', 'Strategy']].sum())
    np.exp(data1[['Returns', 'Strategy']].std() * 252**0.5)
    ax = data1[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))
    data1['Position'].plot(ax=ax, secondary_y='Position', style='--')
    ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
    plt.show()


    sma1 = range(2, 65, 2)
    sma2 = range(10, 283, 5)
    results = pd.DataFrame()
    for SMA1, SMA2 in product(sma1, sma2):
        data1 = pd.DataFrame(raw[symbol])
        data1.dropna(inplace=True)
        data1['Returns'] = np.log(data1[symbol] / data1[symbol].shift(1))
        data1['SMA1'] = data1[symbol].rolling(SMA1).mean()             
        data1['SMA2'] = data1[symbol].rolling(SMA2).mean()             
        data1.dropna(inplace=True)             
        data1['Position'] = np.where(data1['SMA1'] > data1['SMA2'], 1, -1)             
        data1['Strategy'] = data1['Position'].shift(1) * data1['Returns']             
        data1.dropna(inplace=True)             
        perf = np.exp(data1[['Returns', 'Strategy']].sum())             
        results = results.append(pd.DataFrame(
            {'SMA1': SMA1, 'SMA2': SMA2,                          
            'MARKET': perf['Returns'],                          
            'STRATEGY': perf['Strategy'],                          
            'OUT': perf['Strategy'] - perf['Returns']},                          
            index=[0]), ignore_index=True)

    results.sort_values('OUT', ascending=False).head(7)
    print(results)


    data = pd.DataFrame(raw[symbol])
    data.columns = [symbol]
    data['returns'] = np.log(data / data.shift(1)) * 100
    data.fillna(0, inplace=True)
    data['direction'] = np.sign(data['returns']).astype(int)

    lags = 2
    def create_lags(data, lags=2):
        global cols
        cols = []
        for lag in range(1, lags + 1):
            col = 'lag_{}'.format(lag)
            data[col] = data['returns'].shift(lag)
            cols.append(col)

    print('\n             \033[4m Histogram of STOCK returns: \033[0m\n')
    data['returns'].hist(bins=35, figsize=(10,5))
    plt.show()


    create_lags(data)
    data.dropna(inplace=True)
    data.plot.scatter(x='lag_1', y='lag_2', c='returns', cmap='coolwarm', figsize=(10,6), colorbar=True)
    plt.axvline(0, c='r', ls='--')
    plt.axhline(0, c='r', ls='--')
    plt.show()


    model = LinearRegression()  
    data['pos_ols_1'] = model.fit(data[cols], data['returns']).predict(data[cols])
    data['pos_ols_2'] = model.fit(data[cols], data['direction']).predict(data[cols])  
    data[['pos_ols_1', 'pos_ols_2']].head()
    data[['pos_ols_1', 'pos_ols_2']] = np.where(data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)  
    data['pos_ols_1'].value_counts()  
    data['pos_ols_2'].value_counts()
    (data['pos_ols_1'].diff() != 0).sum()  
    (data['pos_ols_2'].diff() != 0).sum()
    data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
    data['strat_ols_2'] = data['pos_ols_2'] * data['returns']
    data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)
    (data['direction'] == data['pos_ols_1']).value_counts()  
    (data['direction'] == data['pos_ols_2']).value_counts()  
    data[['lag_1', 'lag_2', 'returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(figsize=(15, 6))
    plt.show()


    model = KMeans(n_clusters=2, random_state=0)
    model.fit(data[cols])
    data['pos_clus'] = model.predict(data[cols])
    data['pos_clus'] = np.where(data['pos_clus'] == 1, -1, 1)
    data['pos_clus'].values
    plt.figure(figsize=(10, 6))
    plt.scatter(data[cols].iloc[:, 0], data[cols].iloc[:, 1], c=data['pos_clus'], cmap='coolwarm')
    plt.show()
    data['strat_clus'] = data['pos_clus'] * data['returns']
    data[['returns', 'strat_clus']].sum().apply(np.exp)
    (data['direction'] == data['pos_clus']).value_counts()
    data[['returns', 'strat_clus']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()

    def create_bins(data, bins=[0]):
        global cols_bin
        cols_bin = []
        for col in cols:
            col_bin = col + '_bin'
            data[col_bin] = np.digitize(data[col], bins=bins)  
            cols_bin.append(col_bin)

    create_bins(data)        
    data[cols_bin + ['direction']].head() 
    grouped = data.groupby(cols_bin + ['direction'])
    grouped.size()
    res = grouped['direction'].size().unstack(fill_value=0) 

    def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: yellow' if v else '' for v in is_max]
    res.style.apply(highlight_max, axis=1)             