
import pandas as pd
import quandl, math, datetime
from datetime import timedelta
import numpy as np
from sklearn import preprocessing, svm
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = quandl.get("EOD/MSFT", authtoken="REPiQ6KMLB2hr9QAP-s7")
df = df[['Open', 'Low','High', 'Close', 'Volume']]

df['HL_PCT'] = df['HL_PCT'] = (df['High'] - df['Low'])/df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open'])/df['Open'] * 100.0
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
df.fillna(-99999, inplace=True)
forecast_col = 'Close'
forecast_out = 10
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = df.drop(['label'], axis = 1).to_numpy()
y = df['label'].to_numpy()

X = preprocessing.scale(X)

prediction_size = 200

X_train = X[:-prediction_size]
y_train = y[:-prediction_size]

X_test = X[-prediction_size:]
y_test = y[-prediction_size:]

regr = svm.SVR(kernel = 'linear')
regr.fit(X_train, y_train)

#forecasting the future with a proven regression method after regr.score()
forecast_set = regr.predict(X_test)

df['Forecast'] = np.nan

#linked list like thinking,,, get last date --> find next date using unix and timestamp
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
print(last_unix)
one_day = 86400
next_unix = last_unix + float(one_day)
#looping thru the forecaseset of length predictionsize and adding that with index(next_date) and other columns being np.nan
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += float(one_day)
    df.append(pd.DataFrame(index = [next_date]))
    df.at[next_date, 'Forecast'] = i
#plotting close and prediction on the same graph as continuation
df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
