import numpy as np
#(Binary) Cross Entropy Loss

def loss(Y, Y_hat):
    #Y is the matrix of labels
    #Y_hat is the prediction, or last activation layer
    #returns a scalar 
    residuals = (Y - Y_hat)
    cross_entropies = -Y * np.log(Y_hat) - (1-Y) * np.log(1 - Y_hat)
    return np.mean(np.nan_to_num(cross_entropies))

def errors(Y, Y_hat, Z):
    #Y and Y_hat as above.
    #Z is the weighted input into the last activation layer
    return Y_hat - Y

