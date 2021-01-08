import yfinance as yf
import numpy as np
import pandas as pd
from pandas.io.pickle import read_pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class LSTM_RNN(object):
    def __init__(self, ticker):
        self.ticker = ticker

    def getData(self):
        self.data = yf.download(self.ticker, start='2010-01-01', end='2018-01-01')
        self.train = self.data.iloc[:int(len(self.data)*.8)]
        self.test = self.data.iloc[int(len(self.data)*.2):]            

    def configure(self): # train, test, feature-scaling
        self.getData()
        self.training_set = self.train.iloc[:, :1].values     
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.training_set_scaled = self.sc.fit_transform(self.training_set)
        self.X_train = []
        self.y_train = []
        for i in range(100, len(self.train)):
            self.X_train.append(self.training_set_scaled[i-100:i, 0])
            self.y_train.append(self.training_set_scaled[i, 0])
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))

    def build_rnn(self):
        self.configure()
            # Initialising the RNN
        self.regressor = Sequential()
            # Adding the first LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (self.X_train.shape[1], 1)))
        self.regressor.add(Dropout(0.2))
            # Adding a second LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units = 50, return_sequences = True))
        self.regressor.add(Dropout(0.2))
            # Adding a third LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units = 50, return_sequences = True))
        self.regressor.add(Dropout(0.2))
             # Adding a fourth LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units = 50))
        self.regressor.add(Dropout(0.2))
            # Adding the output layer
        self.regressor.add(Dense(units = 1))
            # print summary
        self.regressor.summary()
            # Compiling the RNN
        self.regressor.compile(optimizer = 'adam', loss=self.loss)
            # Fitting the RNN to the Training set
        self.regressor.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size)
        loss = self.regressor.evaluate(self.X_train, self.y_train, verbose=0)
        L = loss * 100
        LOSS = "%{:,.4f}".format(L)
        print('\n Loss = ', LOSS, '\n')

    def predict_rnn(self):
        self.build_rnn()
        self.real_stock_price = self.test.iloc[:, :1].values
            # Getting the predicted stock price of 2017
        self.dataset_total = pd.concat((self.train['Open'], self.test['Open']), axis = 0)
        self.inputs = self.dataset_total[len(self.dataset_total) - len(self.test) - 100:].values
        self.inputs = self.inputs.reshape(-1,1)
        self.inputs = self.sc.transform(self.inputs)
        self.X_test = []
        for i in range(100, len(self.test)):
            self.X_test.append(self.inputs[i-100:i, 0])
        self.X_test = np.array(self.X_test)
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        self.predicted_stock_price = self.regressor.predict(self.X_test)
        self.predicted_stock_price = self.sc.inverse_transform(self.predicted_stock_price)	

    def viz(self, epochs, batch_size, loss = 'mean_squared_error'):    # Visualising the results
        sp = yf.download('^GSPC', start='2010-01-01', end='2019-01-01')['Adj Close']
        sp.columns = ['sp500']
        self.epochs, self.batch_size, self.loss = epochs, batch_size, loss
        self.predict_rnn()
        plt.plot(self.real_stock_price, color = 'red', label = f'Real {self.ticker} Stock Price')
        plt.plot(self.predicted_stock_price, color = 'blue', label = f'Predicted {self.ticker} Stock Price')
        plt.plot(sp, lw=1,label='sp500-actual', color='g', ls='--')
        plt.title(f'{self.ticker} Stock Price Prediction:')
        plt.ylabel(f'{self.ticker} Stock Price')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # path = '/home/gordon/work/Where-To-Put-Your-Money-In-Hard-Times/data/raw/'
    LSTM_RNN('^GSPC').viz(epochs = 5, batch_size = 50, loss = 'mean_squared_error')