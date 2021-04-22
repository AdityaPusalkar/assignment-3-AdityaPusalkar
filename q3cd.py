import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler    
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold     
from metrics import *
import math, copy

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
best_acc = 0
best_LR = None
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
    if(acc>best_acc):
        best_acc = acc
        best_LR = copy.deepcopy(LR)
    print(f'Accuracy for Fold {i}:', acc)
    ov_ac+=acc
    i+=1
print("Overall Accuracy:", ov_ac/4)

y_hat = best_LR.predict_multiclass(X)
confusion_matrix = np.zeros((10,10))
for k in range(len(y)):
    confusion_matrix[y[k]][y_hat[k]]+=1
print("Confusion Matrix")
print(confusion_matrix)
print('Accuracy:', accuracy(y_hat, y))
for k in range(10):
    p = precision(y_hat, y, k)
    r = recall(y_hat, y, k)
    print(f"Precision of class {k}:", p)
    print(f"Recall of class {k}:", r)
    print(f"F-Score of class {k}:", 2*r*p/(r+p))

pca = PCA(n_components=2)
X = pca.fit_transform(load_digits().data)
plt.scatter(X[:,0],X[:,1], c=load_digits().target, cmap="Paired")
plt.colorbar()
plt.show()