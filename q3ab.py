import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler         
from metrics import *
import math

scalar = MinMaxScaler()
data = load_digits()

X = pd.DataFrame(data['data'])
y = pd.Series(data['target'])

scalar.fit(X)
X = scalar.transform(X)
X = pd.DataFrame(X) # This scales data to the range 0-1 and is easier to train

LR = LogisticRegression()
LR.fit_multiclass(X,y,n_iter=60, lr=6e-2)
y_hat = LR.predict_multiclass(X)
print('Accuracy: ', accuracy(y_hat, y))

LR = LogisticRegression()
LR.fit_multiclass_autograd(X,y,n_iter=60, lr=1e-3)
y_hat = LR.predict_multiclass(X)
print('Accuracy: ', accuracy(y_hat, y))