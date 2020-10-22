"""
# 多层前馈神经网络模型
# 五矿稀土，科大讯飞，北方稀土，君正集团，中顺洁柔
# 以收盘价作为判断的标准
# 数据从20150101到20200930
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

import csv
import random


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def normalization(a_list):
    max_num = max(a_list)
    min_num = min(a_list)
    for key, num in enumerate(a_list):
        a_list[key] = (num - min_num) / (max_num - min_num)


beifang_riseOrDrop = []  # final result
beifang_data = []  # 收盘价
beifang_openPrice = []
beifang_highestPrice = []
beifang_lowestPrice = []
beifang_closestPrice = []
beifang_turnOverValue = []
beifang_dealAmount = []
beifang_turnOverRate = []
beifang_marketValue = []
beifang_chgPct = []
beifang_PE = []
beifang_PB = []
beifang_vwap = []
with open(file="beifang20191001.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        beifang_data.append(float(row[11]))
        beifang_openPrice.append(float(row[8]))
        beifang_highestPrice.append(float(row[9]))
        beifang_lowestPrice.append(float(row[10]))
        beifang_turnOverValue.append(float(row[13]))
        beifang_dealAmount.append(int(row[14]))
        beifang_turnOverRate.append(float(row[15]))
        beifang_marketValue.append(float(row[18]))
        beifang_chgPct.append(float(row[19]))
        beifang_PE.append(float(row[20]))
        beifang_PB.append(float(row[22]))
        beifang_vwap.append(float(row[24]))
# beifang_riseOrDrop中如果等于1，说明相较于前一天的收盘价来说是增长的，反之，则说明相较于前一天的收盘价来说是减少的
for i in range(1, len(beifang_data)):
    if beifang_data[i] >= beifang_data[i - 1]:
        beifang_riseOrDrop.append(1)
    else:
        beifang_riseOrDrop.append(0)
beifang_openPrice.pop()
beifang_highestPrice.pop()
beifang_lowestPrice.pop()
beifang_turnOverValue.pop()
beifang_dealAmount.pop()
beifang_turnOverRate.pop()
beifang_marketValue.pop()
beifang_chgPct.pop()
beifang_PE.pop()
beifang_PB.pop()
beifang_vwap.pop()
normalization(beifang_openPrice)
normalization(beifang_highestPrice)
normalization(beifang_lowestPrice)
normalization(beifang_turnOverValue)
normalization(beifang_dealAmount)
normalization(beifang_turnOverRate)
normalization(beifang_marketValue)
normalization(beifang_chgPct)
normalization(beifang_PE)
normalization(beifang_PB)
normalization(beifang_vwap)

# input_data = [beifang_openPrice, beifang_highestPrice, beifang_lowestPrice, beifang_turnOverValue, beifang_dealAmount,
#               beifang_turnOverRate, beifang_marketValue, beifang_chgPct, beifang_PE,beifang_PB, beifang_vwap]
# X为(10,1401),Y为(1,1401)
# X = np.array([beifang_openPrice, beifang_highestPrice, beifang_lowestPrice, beifang_turnOverValue, beifang_dealAmount,
#               beifang_turnOverRate, beifang_marketValue, beifang_chgPct, beifang_PE, beifang_vwap])
X = np.array([beifang_openPrice, beifang_highestPrice, beifang_lowestPrice, beifang_turnOverValue, beifang_dealAmount,
              beifang_turnOverRate, beifang_marketValue, beifang_chgPct, beifang_PE, beifang_PB, beifang_vwap])
Y = np.array([beifang_riseOrDrop])
print("X.shape = ", X.shape)
print("Y.shape = ", Y.shape)


# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -(1.0 / m) * np.sum(logprobs)

    cost = np.squeeze(cost)
    assert (isinstance(cost, float))

    return cost


# GRADED FUNCTION: backward_propagation

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1.0 / m * np.dot(dZ2, A1.T)
    db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1.0 / m * np.dot(dZ1, X.T)
    db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=1.2)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        # print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions


# n_x, n_h, n_y = layer_sizes(X, Y)
# parameters = initialize_parameters(n_x, n_h, n_y)
# grads = backward_propagation(parameters, forward_propagation(X, parameters)[1], X, Y)
# update_parameters(parameters,grads)
# predictions = predict(parameters, )
# print("predictions mean = " + str(np.mean(predictions)))
parameters = nn_model(X, Y, 10)
W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]

Z1 = np.dot(W1, X) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)

hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)

    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
