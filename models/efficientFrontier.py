import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set_palette('cubehelix')
# plt.style.use('seaborn-colorblind') #alternative
plt.rcParams['figure.figsize'] = [8, 4.5]
plt.rcParams['figure.dpi'] = 300
import warnings
warnings.filterwarnings('ignore')

class Efficient_Frontier(object):
    def __init__(self, N_PORTFOLIOS, N_DAYS, RISKY_ASSETS, START_DATE, END_DATE, n_assets):
        self.N_PORTFOLIOS = N_PORTFOLIOS
        self.N_DAYS = N_DAYS
        self.RISKY_ASSETS = RISKY_ASSETS
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE
        self.n_assets = n_assets

    def getData_graph(self):
        self.prices_df = yf.download(self.RISKY_ASSETS, start=self.START_DATE, end=self.END_DATE, adjusted=True)
        print(f'Downloaded {self.prices_df.shape[0]} rows of data.')
        
    def annualized_return(self):
        self.getData_graph()
        self.returns_df = self.prices_df['Adj Close'].pct_change().dropna()
        self.avg_returns = self.returns_df.mean() * self.N_DAYS
        self.cov_mat = self.returns_df.cov() * self.N_DAYS

    def ef_setup(self):
        self.annualized_return()
            # simulate random portfolio weights:
        np.random.seed(42)
        self.weights = np.random.random(size=(self.N_PORTFOLIOS, self.n_assets))
        self.weights /=  np.sum(self.weights, axis=1)[:, np.newaxis]
            # calculate portfolio metrics:
        self.portf_rtns = np.dot(self.weights, self.avg_returns)
        self.portf_vol = []
        for i in range(0, len(self.weights)):
            self.portf_vol.append(np.sqrt(np.dot(self.weights[i].T, np.dot(self.cov_mat, self.weights[i]))))
        self.portf_vol = np.array(self.portf_vol)  
        self.portf_sharpe_ratio = self.portf_rtns / self.portf_vol
            # create joint dataframe with all data:
        self.portf_results_df = pd.DataFrame(
            {'returns': self.portf_rtns, 'volatility': self.portf_vol, 'sharpe_ratio': self.portf_sharpe_ratio})
            # locate points creating efficient frontier:
        self.N_POINTS = 100
        self.portf_vol_ef = []
        self.indices_to_skip = []
        self.portf_rtns_ef = np.linspace(self.portf_results_df.returns.min(), self.portf_results_df.returns.max(), self.N_POINTS)
        self.portf_rtns_ef = np.round(self.portf_rtns_ef, 2)    
        self.portf_rtns = np.round(self.portf_rtns, 2)
        for point_index in range(self.N_POINTS):
            if self.portf_rtns_ef[point_index] not in self.portf_rtns:
                self.indices_to_skip.append(point_index)
                continue
            self.matched_ind = np.where(self.portf_rtns == self.portf_rtns_ef[point_index])
            self.portf_vol_ef.append(np.min(self.portf_vol[self.matched_ind]))
        self.portf_rtns_ef = np.delete(self.portf_rtns_ef, self.indices_to_skip)

    def plot_efficientFrontier(self):
        self.ef_setup()
            # plot efficient frontier:
        MARKS = ['o', 'X', 'd', '*', '+']
        fig, ax = plt.subplots()
        self.portf_results_df.plot(
            kind='scatter', x='volatility', y='returns', c='sharpe_ratio',cmap='RdYlGn', edgecolors='black', ax=ax)
        ax.set(xlabel='Volatility', ylabel='Expected Returns', title='Efficient Frontier')
        ax.plot(self.portf_vol_ef, self.portf_rtns_ef, 'b--')
        for asset_index in range(self.n_assets):
            ax.scatter(x=np.sqrt(
                self.cov_mat.iloc[asset_index, asset_index]),
                y=self.avg_returns[asset_index], 
                marker=MARKS[asset_index], 
                s=150, 
                color='black',
                label=self.RISKY_ASSETS[asset_index])
        ax.legend()
        plt.tight_layout()
        plt.show()

    def results_maxSharpeRatio(self):
        self.plot_efficientFrontier()
        self.max_sharpe_ind = np.argmax(self.portf_results_df.sharpe_ratio)
        self.max_sharpe_portf = self.portf_results_df.loc[self.max_sharpe_ind]
        self.min_vol_ind = np.argmin(self.portf_results_df.volatility)
        self.min_vol_portf = self.portf_results_df.loc[self.min_vol_ind]
        print('Maximum Sharpe Ratio portfolio ----')
        print('Performance:')
        for index, value in self.max_sharpe_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(self.RISKY_ASSETS, self.weights[np.argmax(self.portf_results_df.sharpe_ratio)]):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)
        print('\n')

    def results_minVolatility(self):
        self.results_maxSharpeRatio()
        print('Minimum Volatility portfolio ----')
        print('Performance:')
        for index, value in self.min_vol_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(self.RISKY_ASSETS, self.weights[np.argmin(self.portf_results_df.volatility)]):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)
        print('\n')

    def final_plot(self):
        self.results_minVolatility()
        fig, ax = plt.subplots()
        self.portf_results_df.plot(
            kind='scatter', x='volatility', y='returns', c='sharpe_ratio',cmap='RdYlGn', edgecolors='black', ax=ax)
        ax.scatter(
            x=self.max_sharpe_portf.volatility, 
            y=self.max_sharpe_portf.returns, c='black', marker='*', s=200, label='Max Sharpe Ratio'
            )
        ax.scatter(
            x=self.min_vol_portf.volatility,
            y=self.min_vol_portf.returns, c='black', marker='P', s=200, label='Min Volatility'
            )
        ax.set(xlabel='Volatility', ylabel='Expected Returns', 
            title='Efficient Frontier')
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    N_PORTFOLIOS = 10 ** 5
    N_DAYS = 252
    RISKY_ASSETS = ['AAPL', 'TSLA', 'AMZN', 'MSFT']
    RISKY_ASSETS.sort()
    START_DATE = '2019-01-01'
    END_DATE = '2020-12-24'
    n_assets = len(RISKY_ASSETS)

    # ef = Efficient_Frontier(N_PORTFOLIOS, N_DAYS, RISKY_ASSETS, START_DATE, END_DATE, n_assets)
    # ef.final_plot()