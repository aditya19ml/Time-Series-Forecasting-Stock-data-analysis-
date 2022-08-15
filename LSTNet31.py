import os
from tabnanny import verbose
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import LSTM_Prep
#from lstnet_model import *
#import tensorflow
from Loss import *
from TCN_model import TCN

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau #Learning rate scheduler for when we reach plateaus
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
 


 
 
 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def LSTNet3(input_shape = (11,1),output_shape=(1), dropout = 0.25,head_size=64,num_heads=4,ff_dim=16):
  inputs = Input(shape=input_shape)
  inputs1 = Input(shape=output_shape)

   #transformer
  # x = transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout)
  # x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)  
  # x1 = Dense(11)(inputs1)
  # x1 = Reshape((1, 11))(x1)
  # y = Dense(10, activation='relu')(x+x1) 

  x = transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout)
  x =  Flatten()(x)
  x = RepeatVector(1)(x)

  x = LSTM(100)(x)
  x = Reshape((-1,100))(x)
  y = TimeDistributed(Dense(11))(x)


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
  model.compile(optimizer = linex_loss_ret, loss = 'mse', metrics =  [RSE, tensorflow.keras.metrics.RootMeanSquaredError()])
  print(model.summary())

  return model

