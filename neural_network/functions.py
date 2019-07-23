import numpy as np

def sigmoid(Z, derivative=False):
    if derivative:
        sig = sigmoid(Z)
        return sig*(1-sig)
    else:
        return 1/(1 + np.exp(-Z))

def relu(Z, derivative=False):
    if derivative:
        Z_ = np.copy(Z)
        Z_[Z_ > 0] = 1
        Z_[Z_ <= 0] = 0
        return Z_
    else:
        return np.maximum(0,Z)

def quadratic_cost(y, y_hat, derivative=False):
    #calculate squared error for a single observation (y) and prediction (y_hat)
    if derivative:
        return (y_hat - y)
    else:
        return 0.5 * (y - y_hat)**2

