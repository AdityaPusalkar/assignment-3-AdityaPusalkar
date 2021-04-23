import numpy as np
import pandas as pd
import multiLayerPerceptron.MLP_layer as layer
from multiLayerPerceptron.MLP_network import MultiLayerPerceptron
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold 
from keras.datasets import mnist
from keras.utils import np_utils
from metrics import *

X = load_digits().data
y = load_digits().target
X = X.reshape(X.shape[0], 1, 64)
X = X.astype('float32')
y = np_utils.to_categorical(y)

# Defining the Network
net = MultiLayerPerceptron()
net.add(layer.FullyConnectedLayer(64, 32))
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))
net.add(layer.FullyConnectedLayer(32, 16))                   
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))
net.add(layer.FullyConnectedLayer(16, 10))                   
net.add(layer.ActivationLayer(layer.sigmoid, layer.sigmoid_prime))

kf = KFold(n_splits=3)

f = 1
ov_acc = 0
net.use(layer.mse, layer.mse_prime)
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    net.fit(X_train, y_train, epochs=40, learning_rate=2e-1)
    out = net.predict(X_test)
    y_true = [np.argmax(i) for i in out]
    out = np.argmax(out,axis=1)
    yhat = [np.argmax(i) for i in y_test]
    acc = accuracy(pd.Series(yhat),pd.Series(y_true))
    print(f"Accuracy for fold {f}:", acc)
    f+=1
    ov_acc += acc

print("Overall Accuracy:", ov_acc/3)

