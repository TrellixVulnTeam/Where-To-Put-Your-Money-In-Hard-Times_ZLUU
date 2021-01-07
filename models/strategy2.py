import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

tic = 'TSLA'
ticker = yf.Ticker(tic)
hist = ticker.history(period='10y')
df = hist[['Open','High','Low','Close','Volume']]
df = df.sort_index(ascending=False)

btc = df[['Close']]
btc['daily_difference'] = btc['Close'].diff()
btc['signal'] = 0.0
btc['signal'] = np.where(btc['daily_difference'] > 0, 1.0, 0.0)
btc['positions'] = btc['signal'].diff()


print('\033[4mFor each day where Close Price = Buy = Red Arrow, & Sell = Green \033[0m')
buys = btc.loc[btc['positions'] == 1]
sells = btc.loc[btc['positions'] == -1]
fig = plt.figure(figsize=(15,5))
plt.plot(btc.index, btc['Close'], color='gray', lw=2., label='close price')
plt.plot(buys.index, btc.loc[buys.index]['Close'], '^', markersize=3, color='g', label='Buy')
plt.plot(sells.index, btc.loc[sells.index]['Close'], 'v', markersize=3, color='r', label='Sell')
plt.ylabel(tic + ' Price')
plt.xlabel('Date')
plt.title('Buy & Sell Signals')
plt.legend(loc=0)
plt.show()