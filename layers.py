import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.activation = activation
        
    def forward(self, inputs):
        self.output = self.activation(np.dot(inputs, self.weights) + self.biases)
    
    def backward(self, output_gradient, learning_rate):
        
        activation_gradient = self.activation.derivative(self.output)
        delta = output_gradient * activation_gradient
        
        self.weights_gradient = np.dot(self.inputs.T, delta)
        self.biases_gradient = np.sum(delta, axis=0, keepdims=True)

        self.weights += self.d_weights * learning_rate
        self.biases += self.d_biases * learning_rate
        
        return np.dot(delta, self.weights.T)