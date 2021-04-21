import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import jax.numpy as np
# from jax import grad

class Neuron():
    def __init__(self, weights):
        self.weights = weights
        self.bias = 0

class HiddenLayer():
    def __init__(self, neurons, neuron_values):
        self.neurons = neurons
        self.neuron_values = neuron_values

class MultiLayerPerceptron():
    def __init__(self, X, y, n, g):
        self.layers = []
        self.num_neurons = n
        self.activations = g
        self.X = np.array(X)
        self.output = np.array(y)

    def create_Neuron(self, n_p):
        neuron = Neuron(np.zeros(n_p))
        # neuron = Neuron(np.ones(n_p)/15)
        return neuron
    
    def create_HiddenLayer(self, n, n_p):
        neurons = []
        neuron_values = []
        for i in range(n):
            neurons.append(self.create_Neuron(n_p))
            neuron_values.append(0)
        hiddenLayer = HiddenLayer(np.array(neurons), np.array(neuron_values))
        return hiddenLayer

    def sigmoid(self, z):
        return (1 /(1 + np.exp(-z)))

    def relu(self, z):
        return np.maximum(0,z)

    def fit(self, n_iter, lr):
        for i in range(n_iter):
            N = len(self.num_neurons)
            samples = len(self.X)
            for k in range(samples):
                # forward pass
                for j in range(N):
                    weights = np.array([k.weights for k in self.layers[j].neurons])
                    biases = np.array([k.bias for k in self.layers[j].neurons])
                    print(sum(self.X[k]))
                    if(j==0):
                        output = np.dot(weights, self.X[k]) + biases
                        if(self.activations[j] == 'sigmoid'):
                            output = self.sigmoid(output)
                        elif(self.activations[j] == 'relu'):
                            output = self.relu(output)
                        self.layers[j].neuron_values = output
                    else:
                        output = np.dot(weights, self.layers[j-1].neuron_values) + biases
                        if(self.activations[j] == 'sigmoid'):
                            output = self.sigmoid(output)
                        elif(self.activations[j] == 'relu'):
                            output = self.relu(output)
                        self.layers[j].neuron_values = output
                    print(output)
                print(self.layers[-1].neuron_values)
                # backward pass

            
    def create_MLP(self, n_iter, lr):
        N = len(self.num_neurons)
        for i in range(N):
            if(i==0):
                Nf = len(self.X[0])
                hL = self.create_HiddenLayer(self.num_neurons[i], Nf)
                self.layers.append(hL)
            else:
                hL = self.create_HiddenLayer(self.num_neurons[i], self.num_neurons[i-1])
                self.layers.append(hL)
        self.fit(n_iter, lr)

    

    # def predict(self, X):

