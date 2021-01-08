import yfinance as yf
from yahoo_fin.stock_info import tickers_dow, tickers_sp500, tickers_nasdaq, tickers_other
import pickle

class Get_Historical_Data(object):
    def __init__(self, path, period='10y', interval='1d'):
        self.period = period
        self.inter = interval
        self.path = path + 'data/'


    def getData_S(self, tic):
        '''
        Import Single Stock Price History via yahoo finance API
        uses:
            period - time frame
            interval - frequency of prices
        '''
        self.tic = tic
        self.ticker = yf.Ticker(self.tic)
        new_path = self.path + 'raw/' + self.tic + '_data_' + self.period + '_' + self.inter + '.pkl'
        hist = self.ticker.history(period=self.period, interval=self.inter)
        hist.to_pickle(new_path)
        print('\n     * * * Historical Price Data Web Scrape Complete * * *\n')


    def getData_M(self, tics, save_name='sample_'):
        '''
        Import Multiple Stock Price History via yahoo finance API
        uses:
            period - time frame
            interval - frequency of prices
        '''
        new_path = self.path + 'raw/' + save_name + 'data_' + self.period + '_' + self.inter + '.pkl'
        hist = yf.download(tics, period=self.period, interval=self.inter)['Adj Close']
        hist.to_pickle(new_path)
        print('\n     * * * Historical Price Data Web Scrape Complete * * *\n')


    def getData_I(self, name, save_name=[
        'dow_ticker_list.pkl', 'sp500_ticker_list.pkl', 'nasdaq_ticker_list.pkl', 'other_ticker_list.pkl'
    ]):
        '''
        Import Index Stock Price History via yahoo finance API
        uses:
            period - time frame
            interval - frequency of prices
        '''        
        names = ['dow','sp500','nasdaq','other']
        self.tics = []
        for r in range(len(names)):
            if names[r] in name:
                new_name = self.path + 'interim/tickers/' + save_name[r]
                open_file = open(new_name, "rb")
                loaded_list = pickle.load(open_file)
                open_file.close()
                self.tics.append(loaded_list)
        for i in range(len(self.tics)):
            s_name = names[i] + '_'
            hist = yf.download(self.tics[i], period=self.period, interval=self.inter)['Adj Close']
            new_path = self.path + 'raw/' + s_name + self.period + '_' + self.inter + '.pkl'
            hist.to_pickle(new_path)
        print('\n     * * * Historical Price Data Web Scrape Complete * * *\n')


    # def getData_I_tics():
    #     dow, other, nasdaq, sp500 = tickers_dow(), tickers_other(), tickers_nasdaq(), tickers_sp500()
    #     index_ticker_lst = [dow, other, nasdaq, sp500]
    #     save_name = ['dow_ticker_list.pkl', 'other_ticker_list.pkl', 'nasdaq_ticker_list.pkl'] # sp500_ticker_list.pkl
    #     for i in range(len(index_ticker_lst)):
    #         open_file = open(save_name[i], "wb")
    #         pickle.dump(index_ticker_lst[i], open_file)
    #         open_file.close()
    #     print('\n     * * * All Ticker Lists Have Been Web-Scraped & Saved * * *\n')


if __name__ == '__main__':
    pass
    # p = '~/work/Where-To-Put-Your-Money-In-Hard-Times/data/'
    # run = Get_Historical_Data(path=p, period='1y', interval='1d')
    # run.getData_S(tic='^GSPC')
    # run.getData_M(tics=['AAPL','AMZN','MSFT','TSLA','NFLX','^GSPC'], save_name='portfolio_data_')