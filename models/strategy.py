import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
my_year_month_fmt = mdates.DateFormatter('%m/%y')

class Strategy1:
    def __init__(self, tic, PATH):
        self.tic, self.path = tic, PATH

    def model(self):
        hist = yf.download(self.tic, period='10y')['Adj Close']
        hist.to_pickle(self.path+'raw/strategy_data.pkl')
        data = pd.read_pickle(self.path+'raw/strategy_data.pkl')
        short_rolling = data.rolling(window=20).mean()
        long_rolling = data.rolling(window=100).mean()
        start_date = '2015-01-01'
        end_date = '2016-12-31'
        plt.plot(data, label='Price')
        plt.plot(long_rolling, label = '100-days SMA')
        plt.plot(short_rolling, label = '20-days SMA')
        plt.legend(loc='best')
        plt.ylabel('Price in $')
        plt.tight_layout()
        plt.show()
      # CONFIGURE EMA
        ema_short = data.ewm(span=20, adjust=False).mean()
        fig, ax = plt.subplots()
        ax.plot(data, label='Price')
        ax.plot(ema_short, label = 'Span 20-days EMA')
        ax.plot(short_rolling, label = '20-days SMA')
        ax.legend(loc='best')
        ax.set_ylabel('Price in $')
        ax.xaxis.set_major_formatter(my_year_month_fmt)
        plt.tight_layout()
        plt.show()
      # CONFIGURE TRADING POSITIONS
        trading_positions_raw = data - ema_short
        trading_positions = trading_positions_raw.apply(np.sign) * 1/3
        trading_positions_final = trading_positions.shift(1)
      # PLOT TRADING POSITIONS
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(data, label='Price')
        ax1.plot(ema_short, label = 'Span 20-days EMA')
        ax1.set_ylabel('$')
        ax1.legend(loc='best')
        ax1.xaxis.set_major_formatter(my_year_month_fmt)
        ax2.plot(trading_positions_final, label='Trading position')
        ax2.set_ylabel('Trading position')
        ax2.xaxis.set_major_formatter(my_year_month_fmt)
        plt.tight_layout()
        plt.show()
      # ANALYSIS OF RETURNS
        asset_log_returns = np.log(data).diff()
        strategy_asset_log_returns = trading_positions_final * asset_log_returns
      # Get the cumulative log-returns per asset
        cum_strategy_asset_log_returns = strategy_asset_log_returns.cumsum()
      # Transform the cumulative log returns to relative returns
        cum_strategy_asset_relative_returns = np.exp(cum_strategy_asset_log_returns) - 1
        fig, (ax1, ax2) = plt.subplots(2, 1)
        # for c in asset_log_returns:
        ax1.plot(cum_strategy_asset_log_returns.index, cum_strategy_asset_log_returns, label=self.tic)
        ax1.set_ylabel('Cumulative log-returns')
        ax1.legend(loc='best')
        ax1.xaxis.set_major_formatter(my_year_month_fmt)
        # for c in asset_log_returns:
        ax2.plot(cum_strategy_asset_relative_returns.index, 100 * cum_strategy_asset_relative_returns, label=self.tic)
        ax2.set_ylabel('Total relative returns (%)')
        ax2.legend(loc='best')
        ax2.xaxis.set_major_formatter(my_year_month_fmt)
        plt.tight_layout()
        plt.show()
      # Total strategy relative returns. This is the exact calculation.
        cum_relative_return_exact = cum_strategy_asset_relative_returns
        cum_strategy_log_return = cum_strategy_asset_log_returns
        cum_relative_return_approx = np.exp(cum_strategy_log_return) - 1
        fig, ax = plt.subplots()
        ax.plot(cum_relative_return_exact * 100, label='Exact')
        ax.plot(cum_relative_return_approx * 100, label='Approximation')
        ax.set_ylabel('Total cumulative relative returns (%)')
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(my_year_month_fmt)
        plt.tight_layout()
        plt.show()

        def print_portfolio_yearly_statistics(portfolio_cumulative_relative_returns, days_per_year=(52*5)):
            total_days_in_simulation = portfolio_cumulative_relative_returns.shape[0]
            number_of_years = total_days_in_simulation / days_per_year
            # The last data point will give us the total portfolio return
            total_portfolio_return = portfolio_cumulative_relative_returns[-1]
            # Average portfolio return assuming compunding of returns
            average_yearly_return = (1 + total_portfolio_return)**(1/number_of_years) - 1
            print('\n   * Total portfolio return is: ' + '{:5.2f}'.format(100*total_portfolio_return) + '%')
            print('   * Average yearly return is: ' + '{:5.2f}'.format(100*average_yearly_return) + '%\n')
        print_portfolio_yearly_statistics(cum_relative_return_exact)

      # Define the weights matrix for the simple buy-and-hold strategy
        DATA = pd.DataFrame(data)
        simple_weights_matrix = pd.DataFrame(1/3, index=DATA.index, columns=DATA.columns)
      # Get the buy-and-hold strategy log returns per asset
        simple_strategy_asset_log_returns = simple_weights_matrix * asset_log_returns
      # Get the cumulative log-returns per asset
        simple_cum_strategy_asset_log_returns = simple_strategy_asset_log_returns
      # Transform the cumulative log returns to relative returns
        simple_cum_strategy_asset_relative_returns = np.exp(simple_cum_strategy_asset_log_returns) - 1
      # Total strategy relative returns. This is the exact calculation.
        simple_cum_relative_return_exact = simple_cum_strategy_asset_relative_returns
        fig, ax = plt.subplots()
        ax.plot(100*cum_relative_return_exact, label='EMA strategy')
        ax.plot(100*simple_cum_relative_return_exact, label='Buy and hold')
        ax.set_ylabel('Total cumulative relative returns (%)')
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(my_year_month_fmt)
        # plt.tight_layout()
        plt.show()
        print_portfolio_yearly_statistics(simple_cum_relative_return_exact)

if __name__ == '__main__':
    PATH = '~/work/Where-To-Put-Your-Money-In-Hard-Times/data/'
    Strategy1('^GSPC', PATH).model()

    # tic = ['^GSPC', 'SPY']
    # run = Strategy1(tic)
    # run.model()    