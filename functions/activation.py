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