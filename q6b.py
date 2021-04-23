import numpy as np
import pandas as pd
import multiLayerPerceptron.MLP_layer as layer
from multiLayerPerceptron.MLP_network import MultiLayerPerceptron
from sklearn.datasets import load_boston
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler  
import math  
from metrics import *

X = load_boston().data
y= load_boston().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
X = X.reshape(X.shape[0], 1, 13)
X = X.astype('float32')

# Defining the Network
net = MultiLayerPerceptron()
net.add(layer.FullyConnectedLayer(13, 40))                
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))
net.add(layer.FullyConnectedLayer(40, 13))                   
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))
net.add(layer.FullyConnectedLayer(13, 1)) 

kf = KFold(n_splits=3)

i = 1
ov_rmse = 0
net.use(layer.mse, layer.mse_prime)
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    net.fit(X_train, y_train, epochs=100, learning_rate=3e-3)
    y_hat = net.predict(X_test)
    rmse_curr = rmse(pd.Series(y_hat),pd.Series(y_test))
    print(f"RMSE error for Fold {i}:", rmse_curr)
    ov_rmse += rmse_curr
    i+=1

print("Overall RMSE:", ov_rmse/3)


