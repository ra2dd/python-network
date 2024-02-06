import numpy as np

def batch1():
    inputs = [
        [1, 2, 3, 2.5],
        [2.0, 3.0, -1.0, 4.0],
        [-1.7, 2.4, 3.7, -0.9],
    ]

    weights = [
        [0.2, 0.8, -0.4, 1.0],
        [0.5, 0.9, 0.3, -0.5],
        [-0.1, -0.4, 0.2, 0.7],
    ]

    biases = [2, 3, 0.5]

    output = np.dot(inputs, np.array(weights).T) + biases
    print(output)


def batch2():
    inputs = [
        [1, 2, 3, 2.5],
        [2.0, 3.0, -1.0, 4.0],
        [-1.7, 2.4, 3.7, -0.9],
    ]

    weights1 = [
        [0.2, 0.8, -0.4, 1.0],
        [0.5, 0.9, 0.3, -0.5],
        [-0.1, -0.4, 0.2, 0.7],
    ]

    weights2 = [
        [0.3, -0.23, 0.7],
        [-0.6, 0.15, -0.28],
        [-0.3, 0.6, -0.31],
    ]

    biases = [2, 3, 0.5]
    biases2 = [-1, 1.5, -0.5]

    layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases
    output_2 = np.dot(layer1_outputs, np.array(weights2).T) + biases2
    print(output_2)