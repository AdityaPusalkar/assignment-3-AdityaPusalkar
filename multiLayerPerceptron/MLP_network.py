from autograd import numpy as np, elementwise_grad

class MultiLayerPerceptron():
    def __init__(self):
        self.listOfLayers = []
        self.loss = None
        self.lossPrime = None

    def add(self, layer):
        self.listOfLayers.append(layer)
    
    def use(self, loss, loss_prime):
        self.loss = loss
        self.lossPrime = loss_prime
    
    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.listOfLayers:
                output = layer.forwardPass(output)
            result.append(output)
        return result 
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.listOfLayers:
                    output = layer.forwardPass(output)
                err += self.loss(output, y_train[j])
                auto = elementwise_grad(self.loss)
                error = auto(output,y_train[j])
                for layer in reversed(self.listOfLayers):
                    error = layer.backwardPass(error, learning_rate)
            err /= samples

     