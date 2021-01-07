import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import yfinance as yf
from yfinance import ticker

class Arima_Model(object):
    def __init__(self, ticker):
        self.ticker = ticker

    def model_arima(self):
        df = yf.download(self.ticker, period='2y', interval='1d')
        df.reset_index(inplace=True)

        plt.figure(figsize=(20,7))
        plt.title(f'{self.ticker} Index - Autocorrelation plot with lag = 3')
        lag_plot(df['Open'], lag=3)
        plt.xlabel("time")
        plt.ylabel("price")        
        plt.show()

        plt.figure(figsize=(20,7))
        plt.plot(df["Date"], df["Close"])
        plt.title(f"{self.ticker} Index Price vs Time")
        plt.xlabel("time")
        plt.ylabel("price")
        plt.show()

        train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
        training_data = train_data['Close'].values
        test_data = test_data['Close'].values

        history = [x for x in training_data]
        model_predictions = []
        N_test_observations = len(test_data)

        for time_point in range(N_test_observations):
            model = ARIMA(history, order=(4,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            true_test_value = test_data[time_point]
            history.append(true_test_value)
        MSE_error = mean_squared_error(test_data, model_predictions)
        print('\nTesting Mean Squared Error is {}'.format(MSE_error),'\n')

        df.set_index('Date', inplace=True)
        test_set_range = df[int(len(df)*0.7):].index
        plt.figure(figsize=(20,7))
        plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
        plt.plot(test_set_range, test_data, color='red', label='Actual Price')
        plt.title(f'{self.ticker} Prices Prediction')
        plt.xlabel('Date')
        plt.ylabel('Prices')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    ticker = '^GSPC'
    run = Arima_Model(ticker)
    run.model_arima()