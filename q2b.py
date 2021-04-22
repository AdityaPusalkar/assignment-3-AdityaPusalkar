import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import KFold       
from metrics import *
import math, copy

scalar = MinMaxScaler()
data = load_breast_cancer()

X = pd.DataFrame(data['data'])
y = pd.Series(data['target'])

scalar.fit(X)
X = scalar.transform(X)
X = pd.DataFrame(X) # This scales data to the range 0-1 and is easier to train

best_accuracies_fold = []
coefs = [0.05*f for f in range(1,31)]
for k in coefs:
    best_acc_fold = 0
    kf = KFold(n_splits=3)
    best_LR = None
    for train_index, test_index in kf.split(X):
        X_train = X.iloc[train_index]
        y_train = y[train_index]
        X_test = X.iloc[test_index]
        y_test = y[test_index]

        X_train.reset_index(drop = True, inplace = True)
        y_train.reset_index(drop = True, inplace = True)
        X_test.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)

        best_accuracy = 0
        for train_index_nested, test_index_nested in kf.split(X_train):
            X_train_nested = X_train.iloc[train_index_nested]
            y_train_nested = y_train[train_index_nested]
            X_test_nested = X_train.iloc[test_index_nested]
            y_test_nested = y_train[test_index_nested]

            X_train_nested.reset_index(drop = True, inplace = True)
            y_train_nested.reset_index(drop = True, inplace = True)
            X_test_nested.reset_index(drop = True, inplace = True)
            y_test_nested.reset_index(drop = True, inplace = True)

            LR = LogisticRegression(reg="L1", reg_coef=k)
            LR.fit_autograd(X_train_nested,y_train_nested,n_iter=75, lr=8e-3)
            y_hat = LR.predict(X_test_nested)
            acc = accuracy(y_hat, y_test_nested)
            if(acc>best_accuracy):
                best_accuracy=acc
                best_LR = copy.deepcopy(LR)

        y_hat = best_LR.predict(X_test)
        acc = accuracy(y_hat, y_test)
        if(acc>best_acc_fold):
            best_acc_fold=acc
    best_accuracies_fold.append([best_acc_fold,k])
        
best_accuracies_fold.sort(reverse=True)
print("L1 Regularisation")
print([best_accuracies_fold[0][0], best_accuracies_fold[1][0], best_accuracies_fold[2][0]])
print([best_accuracies_fold[0][1], best_accuracies_fold[1][1], best_accuracies_fold[2][1]])

X = pd.DataFrame(data['data'])
y = pd.Series(data['target'])

scalar.fit(X)
X = scalar.transform(X)
X = pd.DataFrame(X) # This scales data to the range 0-1 and is easier to train

best_accuracies_fold = []
coefs = [0.05*f for f in range(1,31)]
for k in coefs:
    best_acc_fold = 0
    kf = KFold(n_splits=3)
    best_LR = None
    for train_index, test_index in kf.split(X):
        X_train = X.iloc[train_index]
        y_train = y[train_index]
        X_test = X.iloc[test_index]
        y_test = y[test_index]

        X_train.reset_index(drop = True, inplace = True)
        y_train.reset_index(drop = True, inplace = True)
        X_test.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)

        best_accuracy = 0
        for train_index_nested, test_index_nested in kf.split(X_train):
            X_train_nested = X_train.iloc[train_index_nested]
            y_train_nested = y_train[train_index_nested]
            X_test_nested = X_train.iloc[test_index_nested]
            y_test_nested = y_train[test_index_nested]

            X_train_nested.reset_index(drop = True, inplace = True)
            y_train_nested.reset_index(drop = True, inplace = True)
            X_test_nested.reset_index(drop = True, inplace = True)
            y_test_nested.reset_index(drop = True, inplace = True)

            LR = LogisticRegression(reg="L2", reg_coef=k)
            LR.fit_autograd(X_train_nested,y_train_nested,n_iter=160, lr=8e-3)
            y_hat = LR.predict(X_test_nested)
            acc = accuracy(y_hat, y_test_nested)
            if(acc>best_accuracy):
                best_accuracy=acc
                best_LR = copy.deepcopy(LR)

        y_hat = best_LR.predict(X_test)
        acc = accuracy(y_hat, y_test)
        if(acc>best_acc_fold):
            best_acc_fold=acc
    best_accuracies_fold.append([best_acc_fold,k])
        
best_accuracies_fold.sort(reverse=True)
print("L2 Regularisation")
print([best_accuracies_fold[0][0], best_accuracies_fold[1][0], best_accuracies_fold[2][0]])
print([best_accuracies_fold[0][1], best_accuracies_fold[1][1], best_accuracies_fold[2][1]])

# Finding important features with L1 regularization
coefs = [0.1*f for f in range(1,36)]
thetas = []
for k in coefs:
    LR = LogisticRegression(reg="L1", reg_coef=k)
    LR.fit_autograd(X,y,n_iter=200, lr=8e-3)
    thetas.append(LR.coef)
    y_hat = LR.predict(X)

print(thetas[-1])

plt.plot(coefs, thetas)
plt.xlabel('lambda')
plt.ylabel('Coefficient Value')
plt.title('Important Features')
plt.show()