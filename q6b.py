import numpy as np
import pandas as pd
import multiLayerPerceptron.MLP_layer as layer
from multiLayerPerceptron.MLP_network import NeuralNetwork
from sklearn.datasets import load_boston
from keras.datasets import mnist
from keras.utils import np_utils
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
net = NeuralNetwork()
net.add(layer.FullyConnectedLayer(13, 26))                
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))
net.add(layer.FullyConnectedLayer(26, 13))                   
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))
net.add(layer.FullyConnectedLayer(13, 1))                   

net.use(layer.mse, layer.mse_prime)
net.fit(X, y, epochs=100, learning_rate=3e-3)

y_hat = net.predict(X)
print("RMSE error:", rmse(pd.Series(y_hat),pd.Series(y)))


