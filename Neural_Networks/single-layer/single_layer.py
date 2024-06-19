import sys

# append the path of the parent directory
sys.path.append("..")

import numpy as np
from functions import activation_functions as act


class Perceptron(object):
    """
    Rosenblatt Perceptron classifier

    This implementation of the Perceptron expects binary class labels
    in {0, 1}.

    Parameters
    ------------
    eta : float (default: 0.1)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Number of passes over the training dataset.
        Prior to each epoch, the dataset is shuffled to prevent cycles.
    """

    def __init__(self, eta=0.1, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):

        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.epochs):
            _errors = 0
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.weights[1:] += update * x_i
                self.weights[0] += update
                _errors += int(update != 0.0)
            self.errors.append(_errors)
        return self

    def net_input(self, X):
        return (np.dot(X, self.weights[1:]) + self.weights[0]).flatten()

    def predict(self, X):
        return act.step(x=self.net_input(X), threshold=0)


class Adaline(object):
    """
    ADALINE classifier

    This implementation of the Adaline expects binary class labels
    in {-1, 1}.

    Parameters
    ------------
    eta : float (default: 0.1)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Number of passes over the training dataset.
        Prior to each epoch, the dataset is shuffled to prevent cycles.
    """

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):

        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.epochs):
            _errors = 0
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.weights[1:] += update * x_i
                self.weights[0] += update
                _errors += int(update != 0.0)
            self.errors.append(_errors)
        return self

    def net_input(self, X):
        return (np.dot(X, self.weights[1:]) + self.weights[0]).flatten()

    def predict(self, X):
        return act.logistic(
            x=self.net_input(X)
        )  # TODO: change this to make a real Adaline
