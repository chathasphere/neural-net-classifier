import math
from sklearn import datasets
import numpy as np
from neural_network import NeuralNetwork

def one_hot(target, n_classes):
    n_samples = target.shape[0]
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), target] = 1
    return one_hot

def transform_data(X,y):
    #assumes X is an ndarray of shape (num_samples, num_features)
    #assumes y is a vector of shape (num_samples,)
    #one-hot encoding of y
    Y = one_hot(y,10)
    #normalizing X
    X = X / X.max()
    #training/test split
    n = math.floor(0.8 * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = indices[:n], indices[n:]
    X_train, X_test = X[train_idx,:], X[test_idx,:]
    Y_train, Y_test = Y[train_idx,:], Y[test_idx,:]
    #zip
    training_data = (X_train, Y_train)
    test_data = (X_test, Y_test)
    return training_data, test_data

def test():
    data = datasets.load_digits()
    X = data.data
    y = data.target
    train, test = transform_data(X,y)
    digits_nn = NeuralNetwork([64,15,10], activations=["sigmoid", "sigmoid"])
    #let's give it a whirl
    digits_nn.train(train, epochs = 300, batch_size = 50,
            learning_rate = 0.5, test_data = test)
    

if __name__ == "__main__":
    test()
