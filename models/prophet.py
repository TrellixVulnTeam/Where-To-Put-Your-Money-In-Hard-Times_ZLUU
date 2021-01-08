import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from fbprophet import Prophet
m = Prophet(daily_seasonality = True)

class Prophet(object):
    def __init__(self, ticker, m, start, end):
        self.ticker = ticker
        self.m = m
        self.start = start
        self.end = end

    def model_prophet(self):
        data = yf.download(self.ticker, start=self.start, end=self.end)
        sp = yf.download('^GSPC',start=self.start, end='2020-01-01')['Adj Close']
        sp.columns = ['sp_actual']
        data.reset_index(inplace=True)
        data.head(5)
      # Select only the important features i.e. the date and price
        data = data[["Date","Close"]] # select Date and Price
      # Rename the features: These names are NEEDED for the model fitting
        data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
        data.head(5)
      # the Prophet class (model) - # fit the model using all data
        self.m.fit(data) 
     # specify the number of days in future
        future = self.m.make_future_dataframe(periods=365) 
        self.prediction = self.m.predict(future)
        self.m.plot(self.prediction)
        plt.plot(sp,lw=1,label='sp500-actual', color='r', ls='--')
        plt.title("Prediction of the Google Stock Price using the Prophet")
        plt.xlabel("Date")
        plt.ylabel("Close Stock Price")
        plt.tight_layout()
        plt.show()

    def extra(self):
      self.m.plot_components(self.prediction)     
      plt.show()

if __name__ == '__main__':
      pass
      # Prophet('^GSPC', m).model_prophet()