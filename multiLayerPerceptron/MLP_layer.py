from autograd import numpy as np, elementwise_grad

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def mse(y_pred, y_true):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class FullyConnectedLayer():
    def __init__(self,input,output):
        self.input = None 
        self.output = None
        self.weights = np.random.rand(input, output) - 0.5
        self.bias = np.random.rand(1, output) - 0.5 
    
    def forwardPass(self, input):
        self.input = input
        self.output = np.dot(self.input,self.weights) + self.bias
        return self.output

    def backwardPass(self, output_error, learningRate):
        inputError = np.dot(output_error,self.weights.T)
        weightsError = np.dot(self.input.T,output_error) 

        self.weights -= learningRate*weightsError 
        self.bias -= learningRate*output_error
        return inputError

class ActivationLayer():
    def __init__(self, activation, activationPrime):
        self.input = None 
        self.output = None
        self.activation = activation 
        self.activationPrime = activationPrime 
    
    def forwardPass(self, input):
        self.input = input 
        self.output = self.activation(self.input)
        return self.output 
    
    def backwardPass(self, output_error, learningRate):
        auto = elementwise_grad(sigmoid)
        return auto(self.input)*output_error 

    
