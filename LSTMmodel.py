import os
from tabnanny import verbose
# import matplotlib
# import matplotlib.pyplot as plt
import pandas as pd
import LSTM_Prep
from lstnet_model import *
import tensorflow
from Loss import *


from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau #Learning rate scheduler for when we reach plateaus
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
 

def LSTMmodel(input_shape, dropout):
    model = Sequential()
    model.add(LSTM(units= 128, return_sequences=True, input_shape=(input_shape)))
    model.add(Dropout(dropout))     
    model.add(Flatten())
    model.add(Dense(1))
  
    model.compile(optimizer =  'Adam', loss = 'mse', metrics = [RRSE, corr])
    print(model.summary())

    return model