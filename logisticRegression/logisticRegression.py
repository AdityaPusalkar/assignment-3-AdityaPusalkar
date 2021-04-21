import pandas as pd
import matplotlib.pyplot as plt
from autograd import grad,elementwise_grad
import autograd.numpy as np
import math

class LogisticRegression():
    def __init__(self, reg=None):
        self.coef = None
        self.X_auto = None
        self.y_auto = None
        self.reg = reg
        self.reg_coef = 0.1
        self.classes = None
        self.loss_function = None

    def sigmoid(self, z):
        return (1 /(1 + np.exp(-z)))

    def error_function(self, theta):
        loss_function = 0
        dot_prod = self.sigmoid(np.dot(theta,self.X_auto))
        loss_function = self.y_auto*dot_prod + (1-self.y_auto)*(1-dot_prod)
        return -np.sum(np.log(loss_function))

    def error_function_L1(self, theta):
        loss_function = 0
        dot_prod = self.sigmoid(np.dot(theta,self.X_auto))
        loss_function = self.y_auto*dot_prod + (1-self.y_auto)*(1-dot_prod)
        return -np.sum(np.log(loss_function)) + self.reg_coef*sum(abs(theta))

    def error_function_L2(self, theta):
        loss_function = 0
        dot_prod = self.sigmoid(np.dot(theta,self.X_auto))
        loss_function = self.y_auto*dot_prod + (1-self.y_auto)*(1-dot_prod)
        return -np.sum(np.log(loss_function)) + self.reg_coef*np.dot(np.transpose(theta),theta)

    def error_function_multiclass(self, theta):
        loss_function = 0
        N = len(self.X_auto)
        Nf = len(self.X_auto[0])
        for i in range(N):
            curr_pred = self.y_auto[i]
            X_theta_exp = np.exp(np.dot(theta, self.X_auto[i]))
            X_theta_sum = np.sum(X_theta_exp)
            X_theta_currnum = X_theta_exp[curr_pred]
            loss_function += np.log(X_theta_currnum/X_theta_sum)
        return loss_function

    def fit(self, X, y, n_iter, lr):
        N = len(X.index)
        Nf = len(X.columns)
        X = np.array(X)
        y = np.array(y)
        curr_coeff = np.zeros(Nf+1)
        for i in range(n_iter):
            for j in range(N):
                curr_X = X[j]
                curr_X = np.insert(np.transpose(curr_X), 0, 1) #Adding extra 1 for the ease of bias calculation of theta
                curr_y = y[j]
                X_theta = np.dot(curr_coeff, curr_X)
                X_diff = self.sigmoid(X_theta) - curr_y
                X_copy = np.empty(len(curr_X))
                X_copy.fill(X_diff)
                errors = np.multiply(X_copy,curr_X)
                curr_coeff -= lr*errors
        self.coef = curr_coeff

    def fit_autograd(self, X, y, n_iter, lr):
        N = len(X.index)
        Nf = len(X.columns)
        X = np.array(X)
        y = np.array(y)
        curr_coeff = np.zeros(Nf+1)
        self.coef = curr_coeff
        for i in range(n_iter):
            curr_X = np.insert(X, 0, 1, axis=1)
            self.X_auto = np.transpose(curr_X)
            self.y_auto = y
            if(self.reg == "L1"):
                mse_auto = elementwise_grad(self.error_function_L1)
            elif(self.reg == "L2"):
                mse_auto = grad(self.error_function_L2)
            else:
                mse_auto = grad(self.error_function)
            dmse = mse_auto(curr_coeff) 
            curr_coeff -= lr*dmse
        self.coef = curr_coeff

    def fit_multiclass(self, X, y, n_iter, lr):
        N = len(X.index)
        Nf = len(X.columns)
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        curr_coeff = np.zeros((num_classes, Nf+1))
        X_copy = np.empty(Nf+1)
        for i in range(n_iter):
            for j in range(N):
                curr_X = X[j]
                curr_X = np.transpose(curr_X)
                curr_X = np.insert(curr_X, 0, 1) #Adding extra 1 for the ease of bias calculation of theta
                curr_y = y[j]
                X_theta_sum = np.sum(np.exp(np.dot(curr_coeff, curr_X)))
                for k in range(num_classes):
                    X_theta_exp = np.exp(np.dot(curr_coeff[k], curr_X))
                    P_k = X_theta_exp/X_theta_sum
                    X_diff = (1 if curr_y==self.classes[k] else 0) - P_k
                    X_copy.fill(X_diff)
                    errors = np.multiply(X_copy,curr_X)
                    curr_coeff[k] += lr*errors
            # print("Progress:",i/n_iter*100)
        self.coef = curr_coeff

    def fit_multiclass_autograd(self, X, y, n_iter, lr):
        N = len(X.index)
        Nf = len(X.columns)
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        curr_coeff = np.zeros((num_classes,Nf+1))
        for i in range(n_iter):
            self.X_auto = np.insert(X, 0, 1, axis=1)
            self.y_auto = y
            mse_auto = grad(self.error_function_multiclass)
            dmse = mse_auto(curr_coeff) 
            curr_coeff += lr*dmse
            # print("Progress:",i/n_iter*100)
        self.coef = curr_coeff

    def predict(self, X):
        y_hat = []
        X = np.array(X)
        N = len(X)
        for i in range(N):
            y_i = self.sigmoid(self.coef[0] + np.dot(self.coef[1:],X[i]))
            if(y_i<0.5):
                y_hat.append(0)
            else:
                y_hat.append(1)
        return(pd.Series(y_hat))

    def predict_multiclass(self, X):
        y_hat = []
        X = np.array(X)
        N = len(X)
        for i in range(N):
            max_pred = 0
            pred_class = self.classes[0]
            for j in range(len(self.classes)):
                pred = self.coef[j][0] + np.dot(self.coef[j][1:],X[i])
                if(pred>max_pred):
                    max_pred=pred
                    pred_class=self.classes[j]
            y_hat.append(pred_class)
        return(pd.Series(y_hat))