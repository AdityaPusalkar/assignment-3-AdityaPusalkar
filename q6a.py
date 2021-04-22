import numpy as np
import pandas as pd
import multiLayerPerceptron.MLP_layer as layer
from multiLayerPerceptron.MLP_network import NeuralNetwork
from sklearn.datasets import load_digits
from keras.datasets import mnist
from keras.utils import np_utils
from metrics import *

X = load_digits().data
y = load_digits().target
X = X.reshape(X.shape[0], 1, 64)
X = X.astype('float32')
y = np_utils.to_categorical(y)

# Defining the Network
net = NeuralNetwork()
net.add(layer.FullyConnectedLayer(64, 32))
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))
net.add(layer.FullyConnectedLayer(32, 16))                   
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))
net.add(layer.FullyConnectedLayer(16, 10))                   
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))

net.use(layer.mse, layer.mse_prime)
net.fit(X, y, epochs=30, learning_rate=2e-1)

out = net.predict(X)
y_true = [np.argmax(i) for i in out]
out = np.argmax(out,axis=1)
yhat = [np.argmax(i) for i in y]
print("Accuracy:", accuracy(pd.Series(yhat),pd.Series(y_true)))

