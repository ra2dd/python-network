import math
import numpy as np

def relu_activ():
    # ReLU activation function
    inputs = [0, 2, -1, 3.3, 2.7, 1.1, 2.2, -100]
    output = []

    # for i in inputs:
    #     if i < 0:
    #         output.append(0)
    #     elif i >= 0:
    #         output.append(i)

    for i in inputs:
        output.append(max(0, i))

    print(output)


def softmax_activ_manual():
    layer_outputs = [4.8, 1.21, 2.385]

    E = math.e

    exp_values = []
    for output in layer_outputs:
        exp_values.append(E ** output)

    print(exp_values)

    norm_base = sum(exp_values)
    norm_values = []

    for value in exp_values:
        norm_values.append(value / norm_base)

    print(norm_values)


def softmax_activ():
    layer_outputs = [4.8, 1.21, 2.385]

    exp_values = np.exp(layer_outputs)
    print(exp_values)

    norm_values = exp_values / np.sum(exp_values)
    print(norm_values)


def softmax_batch():
    layer_outputs = [[4.8, 1.21, 2.385],
                     [8.9, -1.81, 0.2],
                     [1.41, 1.051, 0.026]]
    
    exp_values = np.exp(layer_outputs)
    print(exp_values)
    
    norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    print(norm_values)