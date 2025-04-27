import numpy as np
from layers import Layer_Dense
import activation

class Model:
    def __init__(self, layers = [], activation = None):
        self.layers = layers
        self.activation = activation 
    
    def train(self, X, y, iterations, learning_rate):
        self.X = X
        self.y = y
        for i in range(iterations):
            pass
        self.output = self.layers[:-1].output
        
    def calculateLoss(self):
        self.loss = np.mean(np.square(self.output - self.y))
        
    def predict(self, X):
        for layer in self.layers:
            layer.forward(X)
            X = self.activation.forward(layer.output)
        self.output = X