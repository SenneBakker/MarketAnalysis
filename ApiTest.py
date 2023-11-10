# written by Senne Bakker
# Started on Nvember 2nd, 2023.


import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import sklearn as sk
from TsModelsTest import *



start = "2022-01-01"
end = '2023-1-01'
tcs = yf.download('TCS',start,end)
infy = yf.download('INFY',start,end)
wipro = yf.download('WIPRO.NS',start,end)


close_tcs = tcs['Close']
cap_tsc = tcs['Volume'].values * close_tcs.values

print(tcs.keys())
# print(close_tcs.index.to_list())
# print(close_tcs.values)
# print(type(close_tcs.index.to_list()))






# my_feature = close_tcs.to_numpy()
print(f'number of data points = {len(close_tcs.values)}')
my_label = close_tcs.values
my_feature = np.arange(0, len(my_label))

# print(f'features: \n {my_feature}')
# print(f'label: \n {my_label}')



learning_rate = 0.1
epochs = 50
my_batch_size = 10


my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
# plot_the_model(trained_weight, trained_bias, my_feature, my_label)
# plot_the_loss_curve(epochs, rmse)

plt.figure()
plt.title('Market cap (prod vol and close)')
# plt.scatter(close_tcs.index.to_list(), cap_tsc)
plt.plot(close_tcs.index.to_list(), cap_tsc)


plt.figure()
plt.title('closing prices')
# plt.scatter(close_tcs.index.to_list(), close_tcs.values)
plt.plot(close_tcs.index.to_list(), close_tcs.values)







plt.show()