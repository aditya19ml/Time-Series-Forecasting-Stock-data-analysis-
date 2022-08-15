# Importation
import os
import numpy as np
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
from LSTNet31 import *


import time

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

# Data
dat = pd.read_csv('MRK_2006-01-01_to_2018-01-01.csv')

#preprocessing the series 
# import pywt
# t = dat["Monthly beer production"]
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
# dat["Monthly beer production"] = x

# print(dat)

split = 0.8
sequence_length = 12

data_prep = LSTM_Prep.Data_Prep(dataset = dat)
rnn_df, validation_df = data_prep.preprocess_rnn(date_colname = 'Date', numeric_colname = 'Close', pred_set_timesteps = 10)


series_prep = LSTM_Prep.Series_Prep(rnn_df =  rnn_df, numeric_colname = 'Close')
window, X_min, X_max = series_prep.make_window(sequence_length = sequence_length, 
                                               train_test_split = split, 
                                               return_original_x = True)

X_train, X_test, y_train, y_test = series_prep.reshape_window(window, train_test_split = split)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(type(y_test))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                 Building the LSTM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau #Learning rate scheduler for when we reach plateaus
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
 

# Reset model if we want to re-train with different splits
def reset_weights(model):
    import keras.backend as K
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)  


# Epochs and validation split
EPOCHS = 201
validation = 0.05
y11 = X_train.reshape((2397, 11))

y21 = []
for i in range(2397):
    y21.append(y11[i][6])
y31 = np.array(y21) 
 

 


#models = [LSTMmodel((59,1), 0.5), LSTNetModel(64, 2, 2,0.5, 128, 12, 3, 32, 12, (32,59,1)), LSTNet2()]
model = LSTNet3() 
start = time.time()

y1 = X_test.reshape((599, 11))

y2 = []
for i in range(599):
    y2.append(y1[i][6])
y3 = np.array(y2)
# History object for plotting our model loss by epoch
history = model.fit(X_train, y_train, epochs = 200, validation_split = validation,
            callbacks = [rlrop], verbose = 2)

t = model.predict(X_test)
t = np.reshape(t, (-1, 1))

y_test = np.reshape(y_test, (-1, 1))

plt.plot(t)
plt.plot(y_test)
plt.title('Beer_production_LSTNet_Transformer')
plt.ylabel('Quantity (Lr.)')
plt.xlabel('Time (Month)')
plt.legend(['actual', 'predicted'], loc='upper left')
plt.show()


model.save('trans.h5')

time.sleep(1)
end = time.time()
print(f"Runtime of the program is {end - start}")

print(model.evaluate(X_test ,y_test))          
# Loss History
# plt.plot(history.history['loss'])

# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# #plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


# plt.plot(history.history['val_loss'])
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()


  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
  #              Predicting the future
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Creating our future object
future = LSTM_Prep.Predict_Future(X_test  = X_test, validation_df = validation_df, lstm_model = model)
# Checking its accuracy on our training set
future.predicted_vs_actual(X_min = X_min, X_max = X_max, numeric_colname = 'Monthly beer production')
# Predicting 'x' timesteps out
future.predict_future(X_min = X_min, X_max = X_max, numeric_colname = 'Monthly beer production', timesteps_to_predict = 12, return_future = True)