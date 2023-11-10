from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
import yfinance as yf
import pandas as pd
import keras
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import time


# #train
# start = "2017-01-01"
# end = '2022-1-01'
#
# #validation
# start_val = "2022-1-01"
# end_val = "2023-1-01"
#
# tcs_train = yf.download('TCS',start,end)
# tcs_val = yf.download('TCS',start_val,end_val)
#

#train
start = "2023-10-23"
end = "2023-10-28"

#validation
start_val = "2023-10-30"
end_val = "2023-11-03"

tcs_train = yf.download('TCS',start,end, interval='1m')
tcs_val = yf.download('TCS',start_val,end_val, interval='1m')

tcs_train=(tcs_train-tcs_train.min())/(tcs_train.max()-tcs_train.min())
tcs_val=(tcs_val-tcs_val.min())/(tcs_val.max()-tcs_val.min())



print('train')
print(tcs_train.corr())
print('validation')
print(tcs_val.corr())


time.sleep(30)

print(tcs_train.keys())

# input = ['Open', 'High', 'Low', 'Volume']
input = ['Open']
dims = len(input)

# Split into input (X) and output (Y) variables
X1 = tcs_train[input]
Y1 = tcs_train['Close']
print(X1.head())

X2 = tcs_val[input]
Y2 = tcs_val['Close']





# Create model
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=dims))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
# Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
# Typically ReLu-based activation are used but since it is performed regression, it is needed a linear activation.
model.add(Dense(1, activation="linear"))

# Compile model: The model is initialized with the Adam optimizer and then it is compiled.
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3, decay=1e-3 / 200))

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200, min_delta=1e-4)

# Fit the model
history = model.fit(X1, Y1, validation_data=(X2, Y2), epochs=350, batch_size=1, verbose=2, callbacks=[es])

# Calculate predictions
PredTestSet = model.predict(X1)
PredValSet = model.predict(X2)

# Save predictions
np.savetxt("trainresults.csv", PredTestSet, delimiter=",")
np.savetxt("valresults.csv", PredValSet, delimiter=",")



# Plot training history
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.title('Training History'),
plt.xlabel('Epoch'),
plt.ylabel('Validation Loss')


# Plot actual vs prediction for training set
plt.figure()
TestResults = np.genfromtxt("trainresults.csv", delimiter=",")
plt.plot(Y1,TestResults,'ro')
plt.plot(Y1, Y1, 'b')
plt.title('Training Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Compute R-Square value for training set
TestR2Value = r2_score(Y1,TestResults)
print("Training Set R-Square=", TestR2Value)

# Plot actual vs prediction for validation set
plt.figure()
ValResults = np.genfromtxt("valresults.csv", delimiter=",")
plt.plot(Y2,ValResults,'ro')
plt.plot(Y2, Y2, 'b')
plt.title('Validation Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Compute R-Square value for validation set
ValR2Value = r2_score(Y2,ValResults)
print("Validation Set R-Square=",ValR2Value)



plt.figure()
plt.title('closing prices')
plt.plot(tcs_train['Close'].index.to_list(), tcs_train['Close'].values)


plt.show()
