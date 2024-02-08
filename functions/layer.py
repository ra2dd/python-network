import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]] 

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        # values closer to 0 get 1, values far from max get close to 0 
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

        # print('dot product input:\n', inputs)
        # print('max out of inputs:\n', np.max(inputs, axis=1, keepdims=True))
        # print('inputs - max:\n', inputs - np.max(inputs, axis=1, keepdims=True))
        # print('exp(inputs - max):\n', np.exp(inputs - np.max(inputs, axis=1, keepdims=True)))
        # print('sum of exp(inputs - max):\n', np.sum(exp_values, axis=1, keepdims=True))
        # print('exp / sum(exp):\n', exp_values / np.sum(exp_values, axis=1, keepdims=True))


def network1():
    # create 300x2 data
    X, y = spiral_data(100, 3)
    
    layer1 = Layer_Dense(2,5)
    activation1 = Activation_ReLU()

    layer1.forward(X)
    print('Data after going through the layer:\n', layer1.output)

    activation1.forward(layer1.output)
    print('Normalized data with activation function:\n', activation1.output)


def network2():
    # create number*3x2 data
    X, y = spiral_data(2, 3)

    layer1 = Layer_Dense(2,3)
    activation1 = Activation_ReLU()

    layer2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    print('starting data:\n', X)
    print('weights layer1:\n', layer1.weights)
    print('Dot product X, weights1:\n', layer1.output)
    print('ReLU activation max(0, input):\n', activation1.output)
    print('weights layer2:\n', layer1.weights)
    print('Dot product ReLU, weights2:\n', layer2.output)
    print('Softmax activation:\n', activation2.output)