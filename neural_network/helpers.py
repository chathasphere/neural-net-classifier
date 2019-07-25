import numpy as np

def sigmoid(Z, derivative=False):
    if derivative:
        sig = sigmoid(Z)
        return sig*(1-sig)
    else:
        return 1. / (1. + np.exp(-Z))

def relu(Z, derivative=False):
    if derivative:
        Z_ = np.copy(Z)
        Z_[Z_ > 0] = 1
        Z_[Z_ <= 0] = 0
        return Z_
    else:
        return np.maximum(0,Z)

def softmax(Z, derivative=False):
    #assumes Z is a matrix (p, n) where
    #p is the number of (output) neurons/classes, n is the number of samples
    if derivative:
        pass
    else:
        exps = np.exp(Z) - np.max(Z, axis=0)
        return exps / np.sum(exps, axis=0)
#ToDo: write a softmax function?
