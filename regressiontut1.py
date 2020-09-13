import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing,svm
from sklearn.svm import SVR

#recieve data
df = quandl.get("EOD/MSFT", authtoken="REPiQ6KMLB2hr9QAP-s7")
#extract useful col
df = df[["Open", "High", "Low", "Close", "Volume"]]
#create new col
df['HL_PCT'] = (df['High'] - df['Low'])/df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open'])/df['Open'] * 100.0
#summarize/extract/shorten data
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
#forecast col named
forecast_col = 'Close'
#filling na, dropna(inplace=True) for otherwise
df.fillna(-99999, inplace=True)
#forcast length, here, ten days into the future
forecast_out = 10
#labeling the forcast column and shifting it 10 days upwards
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)
# capital x is used for features(given data)
X = df.drop(['label'],axis=1).values
#lower case y is used for label(the prediction)
y = df['label'].values

#scaling the x
X = preprocessing.scale(X)

test_size = 200

X_train = X[:-test_size]
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]

regr = svm.SVR(kernel = 'linear')
regr.fit(X_train, y_train)

for X,y in zip(X_test,y_test):
    print(f"Model: {regr.predict([X])[0]}, Actual: {y}")
