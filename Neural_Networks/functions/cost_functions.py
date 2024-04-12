import numpy as np

def sum_of_squared_errors(y, y_hat):
    """Calculates the sum of squared errors between the predictions and the real data.

    Args:
        y (_type_): _description_
        y_hat (_type_): _description_

    Returns:
        _type_: _description_
    """
    sse = (np.sum(y - y_hat)^2) / 2
    return sse
    