import pandas
import numpy as np
from pandas.io.pickle import read_pickle

class Look_At_Optimized_Portfolios(object):
    def __init__(self, key):
        path = '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/processed/'
        self.path1 = (path + key + '_max_sharpeRatio_allocation.pkl')
        self.path2 = (path + key + '_min_volatility_allocation.pkl')

    def viz(self):
        df = read_pickle(self.path1)
        fd = read_pickle(self.path2)
        a = np.where(df.T['allocation'] > 0.0)
        b = np.where(fd.T['allocation'] > 0.0)
        return df.T, fd.T, a, b



if __name__ == '__main__':
    # saveName = 'sample_data' 'sp500' 'dow' 'nasdaq' 'sample'
    # saveName = 'broker_pos_data' 'roth_pos_data', 'moveOn_pos_data', 'potential_pos_data'    
    key = 'dow'
    x = Look_At_Optimized_Portfolios(key)
    df, fd, a, b = x.viz()
    print('\nMaximum Sharpe Ratio Portfolio:')
    df.iloc[a]
    print('\n\nMinimum Volatility Portfolio')
    fd.iloc[b]