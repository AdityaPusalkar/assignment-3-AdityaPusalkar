import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler         
from metrics import *
import math

scalar = MinMaxScaler()
data = load_breast_cancer()

X = pd.DataFrame(data['data'])
y = pd.Series(data['target'])

scalar.fit(X)
X = scalar.transform(X)
X = pd.DataFrame(X) # This scales data to the range 0-1 and is easier to train

LR = LogisticRegression()
LR.fit(X,y,n_iter=500, lr=5e-3)
y_hat = LR.predict(X)
print('Accuracy: ', accuracy(y_hat, y))

LR = LogisticRegression()
LR.fit_autograd(X,y,n_iter=400, lr=8e-3)
y_hat = LR.predict(X)
print('Accuracy: ', accuracy(y_hat, y))