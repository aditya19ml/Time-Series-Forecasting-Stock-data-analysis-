import os
from tabnanny import verbose
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import LSTM_Prep
from lstnet_model import *
import tensorflow
from Loss import *
from TCN_model import TCN

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau #Learning rate scheduler for when we reach plateaus
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
import tensorflow

 


def LSTNet2(input_shape = (12,1),dropout = 0.25,head_size=64,num_heads=4,f_dim=16):
  inputs = Input(shape=input_shape)

  #CNN
  x1 = Conv1D(64, 2, padding="same", activation="relu")(inputs)
  #x1 = Conv1D(64, 2, padding="same", activation="relu")(x1)
  #x1 = Conv1D(64, 2, padding="same", activation="relu")(x1)
  #x1 = Dropout(0.5)(x1)
  #x1 = Conv1D(128, 2, padding="same", activation="relu")(x1)
  x1 = Dropout(0.5)(x1)
  
  x1 = MaxPooling1D(pool_size=2)(x1)
  x1 = Flatten()(x1)
  y = Dense(10, activation='relu')(x1)
  y = Reshape((-1,10))(y)

  #LSTM Attention
#   x1 = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet' )
 # x1_outputs  = x1(y) 
  x2 = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet' )
  x2_outputs  = x2(y)   
#   x2 = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet' )
#   x2_outputs  = x2(x1_outputs)  
#   x2 = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet' )
#   x2_outputs  = x2(x1_outputs)  
  x2_outputs = Flatten()(x2_outputs)
  x = Dense(10)(x2_outputs)
  x = Reshape((-1, 10))(x)

  #AR 
  z = Flatten()(inputs)
  z = Dense(10,activation = 'linear')(z)
  z = Reshape((-1, 10))(z)
 

  #concatenate
  a = Concatenate()([x, z])
  a = Flatten()(a)
  b = Dense(10, activation = 'relu')(a)
  outputs = Dense(1)(b)

  model = Model(inputs, outputs)
  model.compile(optimizer = 'Adam', loss = 'mse', metrics = [RRSE, corr] )
  print(model.summary())

  return model