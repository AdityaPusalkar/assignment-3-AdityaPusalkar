import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import StratifiedKFold     
from metrics import *
import math

scalar = MinMaxScaler()
data = load_digits()

X = pd.DataFrame(data['data'])
y = pd.Series(data['target'])

scalar.fit(X)
X = scalar.transform(X)
X = pd.DataFrame(X) # This scales data to the range 0-1 and is easier to train

skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(X, y)

i = 1
ov_ac = 0
for train_index, test_index in skf.split(X, y):
    X_train = X.iloc[train_index]
    y_train = y[train_index]
    X_test = X.iloc[test_index]
    y_test = y[test_index]

    X_train.reset_index(drop = True, inplace = True)
    y_train.reset_index(drop = True, inplace = True)
    X_test.reset_index(drop = True, inplace = True)
    y_test.reset_index(drop = True, inplace = True)

    LR = LogisticRegression()
    LR.fit_multiclass(X_train,y_train,n_iter=50, lr=9e-3) # 50, 9e-3
    y_hat = LR.predict_multiclass(X_test)
    acc = accuracy(y_hat, y_test)
    confusion_matrix = np.zeros((10,10))
    for k in range(len(y_test)):
        confusion_matrix[y_test[k]][y_hat[k]]+=1
    print("Confusion Matrix")
    print(confusion_matrix)
    print(f'Accuracy for Fold {i}:', acc)
    ov_ac+=acc
    i+=1
print("Overall Accuracy:", ov_ac/4)