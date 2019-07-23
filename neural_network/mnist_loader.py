import numpy as np
import pickle, gzip

def one_hot(target, n_classes):
    n_samples = target.shape[0]
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), target] = 1
    return one_hot

def load_mnist_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def extract():
    training, validation, test = load_mnist_data()
    X_train, Y_train = training
    X_test, Y_test = test
    return (X_train, one_hot(Y_train, 10)), (X_test, one_hot(Y_test, 10))

