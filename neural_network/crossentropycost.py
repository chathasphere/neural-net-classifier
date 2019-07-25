import numpy as np
from .helpers import sigmoid

#AKA L2 Error, Mean Square Error
#Dependent upon sigmoid function
#This may be something to clean up eventually

def loss(Y, Y_hat):
    #Y is the matrix of labels
    #Y_hat is the prediction, or last activation layer
    #returns a scalar value, Mean Square Error
    residuals = 0.5 * (Y - Y_hat)**2
    n_samples = Y.shape[0]
    return np.sum(np.linalg.norm(residuals, axis=0)) / n_samples

def errors(Y, Y_hat, Z):
    #Y and Y_hat as above.
    #Z is the weighted input into the last activation layer
    return (Y_hat - Y) * sigmoid(Z, derivative = True)

