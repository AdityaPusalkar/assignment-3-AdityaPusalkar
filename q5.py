import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiLayerPerceptron.multiLayerPerceptron import MultiLayerPerceptron
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

MLP = MultiLayerPerceptron(X,y,[30,30,10],['sigmoid','sigmoid','sigmoid'])
MLP.create_MLP(n_iter=1, lr=6e-2)