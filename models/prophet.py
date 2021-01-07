import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from fbprophet import Prophet

m = Prophet(daily_seasonality = True)

class Prophet(object):
    def __init__(self, ticker):
        self.ticker = ticker

    def model_prophet(self):
        data = yf.download(self.ticker, period='5y')
        data.reset_index(inplace=True)
        data.head(5)

    # Select only the important features i.e. the date and price
        data = data[["Date","Close"]] # select Date and Price
    # Rename the features: These names are NEEDED for the model fitting
        data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
        data.head(5)
    # the Prophet class (model) - # fit the model using all data
         
        m.fit(data) 
    # specify the number of days in future
        future = m.make_future_dataframe(periods=365) 
        prediction = m.predict(future)
        m.plot(prediction)
        plt.title("Prediction of the Google Stock Price using the Prophet")
        plt.xlabel("Date")
        plt.ylabel("Close Stock Price")
        plt.tight_layout()
        plt.show()

        m.plot_components(prediction)     
        plt.show()

if __name__ == '__main__':
    ticker = '^GSPC'
    # run = Prophet(ticker)
    # run.model_prophet()