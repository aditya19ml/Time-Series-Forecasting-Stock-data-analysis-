# Importation
import os
from tabnanny import verbose
#import matplotlib
#import matplotlib.pyplot as plt
import pandas as pd
import LSTM_Prep
import tensorflow
from Loss import *
#from lstnet_model import *
from LSTMmodel import *
from TCN_model import *
from LSTNet1 import *
from LSTNet2 import *
from LSTNet3 import *
from sklearn.model_selection import KFold

 

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

# Datas
dat = pd.read_csv('monthly-beer-production-in-austr (1).csv')

split = 0.8
sequence_length = 13

data_prep = LSTM_Prep.Data_Prep(dataset = dat)
rnn_df, validation_df = data_prep.preprocess_rnn(date_colname = 'Date', numeric_colname = 'Close', pred_set_timesteps = 10)


series_prep = LSTM_Prep.Series_Prep(rnn_df =  rnn_df, numeric_colname = 'Close')
window, X_min, X_max = series_prep.make_window(sequence_length = sequence_length, 
                                               train_test_split = split, 
                                               return_original_x = True)

X_train, X_test, y_train, y_test = series_prep.reshape_window(window, train_test_split = split)


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

 

 


#models = [LSTMmodel((12,1), 0.5), LSTNetModel(64, 2, 2,0.5, 128, 12, 3, 32, 12, (32,12,1)), LSTNet2(), LSTNet3()]
import time


 
start = time.time()
model = LSTNet2()



# History object for plotting our model loss by epoch


# x=X_train
# y=y_train
# kf=KFold(int(len(X_train)/12),shuffle=False)
# fold=0
# i = 0
# for train,test in kf.split(x):
#     # if(i==10):
#     #     break
#     fold+=1
#     X_train=x[train]
#     Y_train=y[train]
#     X_test1=x[test]
#     Y_test1=y[test]
#     history = model.fit(X_train,y_train, epochs = 200, validation_split = validation,
#             callbacks = [rlrop], verbose = 2)
#     #score=model.evaluate(tensorflow.constant(X_test1,dtype=tensorflow.float16),tensorflow.constant(Y_test1,dtype=tensorflow.float16))
#     #print(f"Final oos Score:{score}")
#     i = i + 1

history = model.fit(X_train,y_train, epochs = 100, validation_split = validation,
            callbacks = [rlrop], verbose = 2)

time.sleep(1)
end = time.time()
print(f"Runtime of the program is {end - start}")
print(model.evaluate(X_test, y_test))   

t = model.predict(X_test)
t = np.reshape(t, (-1, 1))

y_test = np.reshape(y_test, (-1, 1))

plt.plot(t, linestyle = 'dashed')
plt.plot(y_test)
#plt.title('MRK_Stock_data_LSTNet-attention')
plt.ylabel('Volume (Lr.)')
plt.xlabel('Time (Month)')
plt.legend(['predicted', 'actual'], loc='upper left')
plt.show()


  


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
future = LSTM_Prep.Predict_Future(X_test  = X_test, validation_df =  rnn_df, lstm_model = model)
# Checking its accuracy on our training set
future.predicted_vs_actual(X_min = X_min, X_max = X_max, numeric_colname = 'Close')
# Predicting 'x' timesteps out
future.predict_future(X_min = X_min, X_max = X_max, numeric_colname = 'Close', timesteps_to_predict = 100, return_future = True)
