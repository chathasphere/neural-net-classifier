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

class QuadraticCost():
    #AKA L2 Error, Mean Square Error
    @staticmethod

    def cost(Y, Y_hat):
        #Y is the matrix of labels
        #Y_hat is the prediction, or last activation layer
        #returns a scalar value, Mean Square Error
        residuals = 0.5 * (Y - Y_hat)**2
        n_samples = Y.shape[0]
        return np.sum(np.linalg.norm(residuals, axis=0)) / n
    
    @staticmethod
    def errors(Y, Y_hat, Z):
        #Y and Y_hat as above.
        #Z is the weighted input into the last activation layer
        return (Y_hat - Y) * sigmoid(Z, derivative = True)

class CrossEntropyCost()
    # Binary cross entropy 

    @staticmethod
    def cost(Y, Y_hat):
        cross_entropies =  -(Y @ np.log(Y_hat).T + (1 - Y) @ np.log(1 - Y_hat.T))
        n_samples = Y.shape[0]
        return np.sum(np.nan_to_num(cross_entropies)) / n

    @staticmethod
    def errors(Y, Y_hat, Z):
        return Y_hat - Y


