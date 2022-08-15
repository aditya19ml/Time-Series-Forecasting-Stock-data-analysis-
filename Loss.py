import os
from tabnanny import verbose
# import matplotlib
# import matplotlib.pyplot as plt
import pandas as pd
import LSTM_Prep
#from lstnet_model import *
import tensorflow
import numpy 


#Loss functions 
def RSE(y_true, y_predicted):
    """
    - y_true: Actual values
    - y_predicted: Predicted values """     
    RSS = tensorflow.reduce_sum(tensorflow.square(y_true - y_predicted))
    o = tensorflow.cast((len(y_true) - 2), dtype = tensorflow.float32)
    rse = tensorflow.sqrt(RSS / o)
    return tensorflow.reduce_mean(rse)


def RRSE(y_true, y_predicted):
    """
    - y_true: Actual values
    - y_predicted: Predicted values  """     
    a = tensorflow.reduce_sum(tensorflow.square(y_true - y_predicted))
    b = tensorflow.reduce_sum(tensorflow.square(y_true - tensorflow.reduce_mean(y_predicted)))
     
    rrse = tensorflow.sqrt(a/b)
    return tensorflow.reduce_mean(rrse)

def corr(y_true, y_pred):
    #
    # This function calculates the correlation between the true and the predicted outputs
    #
    num1 = y_true -  tensorflow.reduce_mean(y_true, axis=0)
    num2 = y_pred -  tensorflow.reduce_mean(y_pred, axis=0)
    
    num  =  tensorflow.reduce_mean(num1 * num2, axis=0)
    den  =  tensorflow.math.reduce_std(y_true, axis=0) * tensorflow.math.reduce_std(y_pred, axis=0)
    
    return  tensorflow.reduce_mean(num / den)


def sign_ae(x, y):
    sign_x = tensorflow.math.sign(x)
    sign_y = tensorflow.math.sign(y)
    delta = x - y
    return sign_x * sign_y * tensorflow.math.abs(delta)
    
    
def linex_loss(delta, a=-1, b=1):
     
    if a!= 0 and b > 0:
        loss = b * (tensorflow.math.exp(a * delta) - a * delta - 1)
        return loss
    else:
        raise ValueError
        
        
def linex_loss_val(y_true, y_pred):
    delta = sign_ae(y_true, y_pred)
    res = linex_loss(delta)
    return tensorflow.reduce_mean(res)
    
    
def linex_loss_ret(y_true, y_pred):
    # y_true = y_true.numpy()
    # y_pred = y_pred.numpy()
    # # y_true = y_true.tolist()
    # # y_pred = y_pred.tolist()
    # diff_true = numpy.diff(y_true)
    # diff_pred = numpy.diff(y_pred)

    # diff_true = tensorflow.convert_to_tensor(y_, dtype = tensorflow.float32)
    # diff_pred = tensorflow.convert_to_tensor(diff_pred, dtype = tensorflow.float32)
    
    delta = sign_ae(y_true, y_pred)
    res = linex_loss(delta)
    return tensorflow.reduce_mean(res)