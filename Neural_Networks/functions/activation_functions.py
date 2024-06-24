import numpy as np


def rectified_linear_unit(x):
    """Simple ReLU function.
    Returns 0 for negative values or the value itself if positive.
    Good for perceptrons.

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return max(0, x)


def step(x, threshold):
    """Step activation function.
    Returns 1 if x is bigger than the threshold, otherwise returns 0.
    The step activation function is typically used in n binary classification problems,
    or in feedforward neural networks.

    Parameters
    ----------
    x : _type_
        _description_
    threshold : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return np.where(x >= threshold, 1, 0)


def logistic(x):
    """logistic function.
    This function passes the input through a logistic function (S-shaped function)
    where the large negative or positive values will tend to produce -1 and 1,
    respectively, and values close to 0 will produce rapidly changing values between
    -1 and 1.
    The logistic activation function is a good choice for multi-class classification
    problems where the goal is to produce a continuous output.

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return 1 / (1 + np.exp(-x))
