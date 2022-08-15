 # Importation
import os
from tabnanny import verbose
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import LSTM_Prep
#import tensorflow
from Loss import *
#from lstnet_model import *
from LSTMmodel import *
from TCN_model import *
from LSTNet1 import *
from LSTNet2 import *
from LSTNet3 import *

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

# Data
dat = pd.read_csv('MRK_2006-01-01_to_2018-01-01.csv')

#preprocessing the series 
# import pywt
# t = dat["Close"]
# for i in range(20):
#     cA, cD = pywt.dwt(t,  wavelet='db2')
#     cD1 = []
#     for i in range(len(cD)):
#         cD1.append(0)
#     t = pywt.idwt(cA, cD1, 'db2')
# t1 = []
# for i in range(len(t)):
#     t1.append(t[i])

# print(len(t1)-1)
# x = pd.Series(t1) 
# dat["Close"] = x

# print(dat)

split = 0.8
sequence_length = 60

data_prep = LSTM_Prep.Data_Prep(dataset = dat)
rnn_df, validation_df = data_prep.preprocess_rnn(date_colname = 'Date', numeric_colname = 'Close', pred_set_timesteps = 10)


series_prep = LSTM_Prep.Series_Prep(rnn_df =  rnn_df, numeric_colname = 'Close')
window, X_min, X_max = series_prep.make_window(sequence_length = sequence_length, 
                                               train_test_split = split, 
                                               return_original_x = True)

X_train, X_test, y_train, y_test = series_prep.reshape_window(window, train_test_split = split)


print(X_test)
print(X_train.shape)
print(y_train.shape)

y1 = X_train.reshape((2358, 59))

y2 = []
for i in range(2358):
    y2.append(y1[i][-30])
y3 = np.array(y2) 

print(y3.shape)
