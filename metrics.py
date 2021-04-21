import math

def accuracy(y_hat, y):
    if(not y_hat.empty):
        assert(y_hat.size == y.size)
        N = y.size
        y, y_hat = list(y), list(y_hat)
        accuracy = sum(y[i] == y_hat[i] for i in range(N))/N
        return accuracy
    return 1

def precision(y_hat, y, cls):
    if(not y_hat.empty):
        N = len(y)
        y, y_hat = list(y), list(y_hat)
        total = sum(y[i] == cls for i in range(N))
        if(total==0):
            return 1
        else:
            correct = sum(y[i] == y_hat[i] and y[i] == cls for i in range(N))
            return correct/total 
    return 1

def recall(y_hat, y, cls):
    if(not y_hat.empty):
        N = len(y)
        y, y_hat = list(y), list(y_hat)
        total = sum(y_hat[i] == cls for i in range(N))
        if(total==0):
            return 1
        else:
            correct = sum(y[i] == y_hat[i] and y[i] == cls for i in range(N))
            return correct/total 
    return 1

def rmse(y_hat, y):
    if(not y_hat.empty):
        assert(y_hat.size == y.size)
        N = len(y)
        y, y_hat = list(y), list(y_hat)
        rmse = math.sqrt(sum((y_hat[i]-y[i])**2 for i in range(N))/N)
        return rmse
    return 1

def mae(y_hat, y):
    if(not y_hat.empty):
        assert(y_hat.size == y.size)
        N = len(y)
        y, y_hat = list(y), list(y_hat)
        mae = sum(abs(y_hat[i]-y[i]) for i in range(N))/N
        return mae
    return 1