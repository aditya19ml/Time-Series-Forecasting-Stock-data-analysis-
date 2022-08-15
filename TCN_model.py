import os
from tabnanny import verbose
# import matplotlib
# import matplotlib.pyplot as plt
import pandas as pd
import LSTM_Prep
#from lstnet_model import *
#import tensorflow
from Loss import *


from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau #Learning rate scheduler for when we reach plateaus
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
 


#TCN model 
import tensorflow.keras.backend as K

def channel_normalization(x):
  max_values = K.max(K.abs(x), 2, keepdims = True)
  out = x/max_values
  return out


def wave_net_activation(x):
  tanh_out = Activation('tanh')(x)
  sigm_out = Activation('sigmoid')(x)
  return multiply([tanh_out, sigm_out])

def residual_block(x, activation, nb_filters, kernel_size, padding, dropout_rate = 0):  
  original_x = x
  conv = Conv1D(filters=nb_filters, kernel_size=kernel_size, padding=padding)(x)
  if activation == 'norm_relu':
    x = Activation('relu')(conv)
    x = Lambda(channel_normalization)(x)
  elif activation == 'wavenet':
    x = wave_net_activation(conv)
  else:
    x = Activation(activation)(conv)
  x = SpatialDropout1D(dropout_rate)(x)
  x = Convolution1D(nb_filters, 1, padding='same')(x)
  res_x = add([original_x, x])
  return res_x, x

def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        # print(f'Updated dilations from {dilations} to {new_dilations} because of backwards compatibility.')
        return new_dilations


class TCN:
    """Creates a TCN layer.
        Args:
            input_layer: A tensor of shape (batch_size, timesteps, input_dim).
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            activation: The activations to use (norm_relu, wavenet, relu...).
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation='norm_relu',
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=True,
                 ):
         
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

        # backwards incompatibility warning.
        # o = tcn.TCN(i, return_sequences=False) =>
        # o = tcn.TCN(return_sequences=False)(i)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' paddings are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(i, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(i)')
            print('Second solution is to pip install keras-tcn==2.1.2 to downgrade.')
            raise Exception()

    def __call__(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32]
        x = inputs
        x = Conv1D(self.nb_filters, 1, padding=self.padding)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x, skip_out = residual_block(x,  self.activation, self.nb_filters,
                                             self.kernel_size, self.padding, self.dropout_rate )
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = add(skip_connections)
        x = Activation('relu')(x)

        if not self.return_sequences:
            output_slice_index = -1
            x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
        return x        


def TCNModel(input_shape, dropout):
  inputs = Input(shape=input_shape)
  x1 = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet' )
  x1_outputs  = x1(inputs) 
  x2 = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet' )
  x2_outputs  = x2(x1_outputs)
  x2_outputs = Flatten()(x2_outputs)
  outputs = Dense(1)(x2_outputs)
  model = Model(inputs, outputs)
  model.compile(optimizer = tensorflow.keras.optimizers.Adam(lr = 0.0001), loss = RSE, metrics = [tensorflow.keras.metrics.RootMeanSquaredError(name='rmse')])
  print(model.summary())
  return model