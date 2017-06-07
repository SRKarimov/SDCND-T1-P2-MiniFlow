# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:03:37 2017

@author: Sergei Karimov
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import *

"""
NOTE: Here we're using an Input node for more than a scalar.
In the case of weights and inputs the value of the Input node is
actually a python list!

In general, there's no restriction on the values that can be passed to an Input node.
"""

def test_1():
    inputs, weights, bias = Input(), Input(), Input()
    f = Linear(inputs, weights, bias)

    feed_dict = {
        inputs: [6, 14, 3],
        weights: [0.5, 0.25, 1.4],
        bias: 2
    }

    graph = topological_sort(feed_dict)
    output = forward_pass(f, graph)

    # should be 12.7 with this example
    print(output)


def test_2():
    X, W, b = Input(), Input(), Input()

    f = Linear(X, W, b)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}

    graph = topological_sort(feed_dict)
    output = forward_pass(f, graph)

    """
    Output should be:
    [[-9., 4.],
    [-9., 4.]]
    """
    print(output)


def test_3():
    X, W, b = Input(), Input(), Input()

    f = Linear(X, W, b)
    g = Sigmoid(f)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}

    graph = topological_sort(feed_dict)
    output = forward_pass(g, graph)

    """
    Output should be:
    [[  1.23394576e-04   9.82013790e-01]
     [  1.23394576e-04   9.82013790e-01]]
    """
    print(output)


def test_4():
    y, a = Input(), Input()
    cost = MSE(y, a)

    y_ = np.array([1, 2, 3])
    a_ = np.array([4.5, 5, 10])

    feed_dict = {y: y_, a: a_}
    graph = topological_sort(feed_dict)
    # forward pass
    forward_pass(graph)

    """
    Expected output

    23.4166666667
    """
    print(cost.value)


def test_5():
    """
    Have fun with the number of epochs!

    Be warned that if you increase them too much,
    the VM will time out :)
    """

    # Load data
    data = load_boston()
    X_ = data['data']
    y_ = data['target']

    # Normalize data
    X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

    n_features = X_.shape[1]
    n_hidden = 10
    W1_ = np.random.randn(n_features, n_hidden)
    b1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden, 1)
    b2_ = np.zeros(1)

    # Neural network
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()

    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    l2 = Linear(s1, W2, b2)
    cost = MSE(y, l2)

    feed_dict = {
        X: X_,
        y: y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_
    }

    epochs = 1000
    # Total number of examples
    m = X_.shape[0]
    batch_size = 11
    steps_per_epoch = m // batch_size

    graph = topological_sort(feed_dict)
    trainables = [W1, b1, W2, b2]

    print("Total number of examples = {}".format(m))

    # Step 4
    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            # Step 1
            # Randomly sample a batch of examples
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

            # Reset value of X and y Inputs
            X.value = X_batch
            y.value = y_batch

            # Step 2
            forward_and_backward(graph)

            # Step 3
            sgd_update(trainables)

            loss += graph[-1].value

        print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))


# test_1()
# test_2()
# test_3()
#test_4()
test_5()
