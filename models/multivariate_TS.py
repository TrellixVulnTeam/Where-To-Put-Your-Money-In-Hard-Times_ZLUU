import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# if gpu_devices:
#     print('Using GPU')
#     tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# else:
    # print('Using CPU')
sns.set_style('whitegrid')
np.random.seed(42)    
results_path = Path('results', 'multivariate_time_series')
if not results_path.exists():
    results_path.mkdir(parents=True)


class Multivariate_TS(object):
    def __init__(self, tics=['UMCSENT', 'IPGMFN']):
        self.tics = tics

    def multivariate(self):
        df = web.DataReader(self.tics, 'fred', '1980', '2021').dropna()
        df.columns = ['sentiment', 'ip']
        print(df.info)
        df_transformed = (pd.DataFrame(
            {'ip': np.log(df.ip).diff(12), 'sentiment': df.sentiment.diff(12)}).dropna())
        df_transformed = df_transformed.apply(minmax_scale)
        def create_multivariate_rnn_data(data, window_size):
            y = data[window_size:]
            n = data.shape[0]
            X = np.stack([data[i: j]
                        for i, j in enumerate(range(window_size, n))], axis=0)
            return X, y
        window_size = 18
        X, y = create_multivariate_rnn_data(df_transformed, window_size=window_size)
        print(X.shape, y.shape)
        print(df_transformed.head())
        test_size = 24
        train_size = X.shape[0]-test_size
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        print(X_train.shape, X_test.shape)
        K.clear_session()
        n_features = output_size = 2
        lstm_units = 12
        dense_units = 6
        rnn = Sequential([
            LSTM(units=lstm_units,
            dropout=.1,
            recurrent_dropout=.1,
            input_shape=(window_size, n_features), 
            name='LSTM',
            return_sequences=False),
            Dense(dense_units, name='FC'),
            Dense(output_size, name='Output')]
            )
        print(rnn.summary())
        rnn.compile(loss='mae', optimizer='RMSProp')
        lstm_path = (results_path / 'lstm.h5').as_posix()
        checkpointer = ModelCheckpoint(
            filepath=lstm_path,
            verbose=1,
            monitor='val_loss',
            mode='min',
            save_best_only=True
            )
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10,
            restore_best_weights=True
            )
        result = rnn.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=20,
            shuffle=False,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, checkpointer],
            verbose=1
            )
        fig, axes = plt.subplots(ncols=3, figsize=(14,4))
        columns={'ip': 'Industrial Production', 'sentiment': 'Sentiment'}
        pd.DataFrame(result.history).plot(ax=axes[0], title='Loss vs Value Loss')
        df.rename(columns=columns).plot(ax=axes[1], title='Original Series')
        df_transformed.rename(columns=columns).plot(ax=axes[2], title='Tansformed Series')
        
        sns.despine()
        fig.tight_layout()


        y_pred = pd.DataFrame(
            rnn.predict(X_test), 
            columns=y_test.columns, 
            index=y_test.index
            )
        print(y_pred.info())
        test_mae = mean_absolute_error(y_pred, y_test)
        print(test_mae)

        fig, axes = plt.subplots(ncols=3, figsize=(17, 4))
        pd.DataFrame(result.history).rename(
            columns={'loss': 'Training','val_loss': 'Validation'}).plot(ax=axes[0], 
            title='Train & Validiation Error'
            )
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MAE')
        col_dict = {'ip': 'Industrial Production', 'sentiment': 'Sentiment'}
        for i, col in enumerate(y_test.columns, 1):
            y_train.loc['2010':, col].plot(
                ax=axes[i], 
                label='training', 
                title=col_dict[col]
                )
            y_test[col].plot(ax=axes[i], label='out-of-sample')
            y_pred[col].plot(ax=axes[i], label='prediction')
            axes[i].set_xlabel('')
        axes[1].set_ylim(.0, .9)
        axes[1].fill_between(x=y_test.index, y1=0.0, y2=0.9, color='grey', alpha=.5)
        axes[2].set_ylim(.0, .9)
        axes[2].fill_between(x=y_test.index, y1=0.0, y2=0.9, color='grey', alpha=.5)
        plt.legend()
        fig.suptitle(
            'Multivariate RNN - Results | Test MAE = {:.4f}'.format(test_mae), fontsize=14)
        sns.despine()
        fig.tight_layout()
        fig.subplots_adjust(top=.85)
        plt.show()

if __name__ == '__main__':
    run = Multivariate_TS()
    run.multivariate()