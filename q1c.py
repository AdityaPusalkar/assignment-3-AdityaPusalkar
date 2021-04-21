import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.preprocessing import MinMaxScaler 
from sklearn.datasets import load_breast_cancer
from metrics import *
import math

np.random.seed(42)
data = load_breast_cancer()
scalar = MinMaxScaler()

X = pd.DataFrame(data['data'])
y = pd.Series(data['target'])

scalar.fit(X)
X = scalar.transform(X)
X = pd.DataFrame(X) # This scales data to the range 0-1 and is easier to train

acc_ov = 0
for i in range(3):
    xti = X.iloc[i*190:min(570,(i+1)*190)]
    yti = y[i*190:min(570,(i+1)*190)]
    xi1 = X.iloc[0:i*190]
    xi2 = X.iloc[min(570,(i+1)*190):570]
    yi1 = y[0:i*190]
    yi2 = y[min(570,(i+1)*190):570]

    xi = pd.concat([xi1,xi2])
    yi = pd.concat([yi1,yi2])
    xi.reset_index(drop = True, inplace = True)
    yi.reset_index(drop = True, inplace = True)
    xti.reset_index(drop = True, inplace = True)
    yti.reset_index(drop = True, inplace = True)

    LR = LogisticRegression()
    LR.fit(xi,yi,n_iter=600, lr=7e-03)
    y_hat = LR.predict(xti)
    acc_curr = accuracy(y_hat,yti)
    print(f'Accuracy Fold {i+1}: ', acc_curr)
    acc_ov += acc_curr

print("Overall Accuracy:",acc_ov/3)
