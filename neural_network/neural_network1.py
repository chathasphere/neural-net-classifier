import numpy as np
from .helpers import sigmoid

class TrainingError(Exception):
    pass

def quadratic_cost(y, y_hat, derivative=False):
    """
    Calculate squared error for a single observation and prediction.

    Args:
        y (numpy array): observed value / label: dimension is (num_classes x 1)
        y_hat (numpy array): predicted value
        derivative (bool): if true, calculate cost derivative with respect to 
        y_hat, otherwise calculate cost
    """
    if derivative:
        return (y_hat - y)
    else:
        return 0.5 * (y - y_hat)**2

class NeuralNetwork:
    """
    Feed-forward neural network classifier using sigmoid activation functions and quadratic cost.

    Attributes:
        sizes (list of int): number of neurons per network layer
        weights (list of numpy arrays): weights for connections between layers; dimension is
            (input_layer_size x output_layer_size)
        biases (list of numpy arrays): biases for connections between layers; dimension is
            (input_layer_size(n) x 1)
        
    """

    def __init__(self, sizes):
        """
        Args:
            sizes (list of int): number of neurons per network layer
        """
        self.sizes = sizes
        self.weights = self.init_weights()
        #no bias needed for first layer
        self.biases = [np.random.randn(s,1) * 0.1 for s in sizes[1:]]

    def init_weights(self):
        """
        Initalize weights with Gaussian values scaled down by the squared size of input layer in order
        to normalize variance.

        Returns:
            weights (list of numpy arrays)
        """
        input_sizes = self.sizes[:-1]
        output_sizes = self.sizes[1:]
        weight_dims = list(zip(output_sizes, input_sizes))
        #at layer l, weight[j][k] is the weight from the 
        # kth neuron in the layer l-1 to the jth neuron in layer l
        weights = [np.random.randn(j,k) for j,k in weight_dims]
        #normalize variance by size of inputs
        weights = [W / (W.shape[1]**(0.5)) for W in weights]
        return weights

    def forward_prop_layer(self, layer_number, A):
        """
        Feed output of previous layer forward through one layer of the 
        neural network. 
        Applies a linear transformation with the layer's weight and bias 
        to obtain a matrix Z, 
        which is fed through a (non-linear) sigmoid function, yielding 
        activation matrix A.
        
        Args:
            layer_number (int): index of layer
            A (numpy array): activated matrix input from previous layer.

        Returns:
            A (numpy array): activated output
            Z (numpy array): weighted (pre-activation) input into activation
                function
        """
        W = self.weights[layer_number]
        b = self.biases[layer_number]
        Z = np.dot(W, A) + b
        A = sigmoid(Z)
        return A,Z

    def feed_forward(self, X):
        """
        Feeds network input data through all layers of the neural network.

        Args:
            X (numpy array): Transposed input data (feature set) of shape (num_features x num_samples)

        Returns:
            A (numpy array): Final output of the neural network, i.e. the prediction (Y_hat). Has
                dimension (num_classes x num_samples)
        """
        A = X
        for l in range(0, len(self.sizes) - 1):
            A,Z = self.forward_prop_layer(l,A)
        return A

    def update_batch(self, X, Y, learning_rate):
        """
        Calculates the cost function gradient (w.r.t to weights and biases in each layer) for samples in
        a minibatch. Uses the gradient (multiplied by a small constant learning rate) to 
        update weights and biases. In other words, this function implements minibatch gradient descent
        toward minimizing overall cost.

        Args:
            X (numpy array): input features for minibatch
            Y (numpy array): output labels for minibatch
            learning_rate (float): small (typically < 0.1) multiplicative constant to help ensure
                gradient descent does not overshoot a minimum.

        """
        m = X.shape[0]
        eta = learning_rate
        del_weights, del_biases = self.backpropagate(X,Y,m)
        #iterate through layers and update weights and biases by subtracting error
        for l in range(len(self.sizes) - 1):
            self.weights[l] -= eta * del_weights[-l-1]
            self.biases[l] -= eta * del_biases[-l-1]

    def backpropagate(self, X, Y, m):
        """
        Backpropagation algorithm to calculate the cost gradient with 
        respect to the network's weights and biases for a given batch.

        Args:
            X (numpy array): input
            Y (numpy array): output
            m (int): batch size

        Returns
            del_weights (list of numpy arrays): list of cost gradients for
                weights in each layer of the network
            del_biases (list of numpy arrays): list of cost gradients for
                biases in each layer of the network

        """
        del_weights = []
        del_biases = []
        #list of weighted inputs into each layer
        zs = []
        #set activation of 0th layer: it's the input!
        A = X.T
        #activated outputs from each layer 
        activations = [A]
        #forward pass
        #store A and Z from each layer
        for l in range(0, len(self.sizes) - 1):
            A,Z = self.forward_prop_layer(l, A)
            activations.append(A)
            zs.append(Z)

        #backward pass: calculate the cost gradient at the last layer of 
        #the neural network and work backwards.

        #Derivative of cost function with respect to *last* activation
        d_cost = quadratic_cost(Y.T, activations[-1], derivative = True)
        #errors is a numpy array of dimension
        #(n_neurons x n_batch_samples)
        #It's the derivative of cost with respect to the weighted input
        #(z) of a layer.
        #errors: positive error means output is too high, 
        #negative error means output is too low
        #In general, the idea is to subtract error from weighted input
        #to correct output.
        errors = d_cost * sigmoid(zs[-1], derivative = True)
        
        #calculate gradient with respect to bias:
        #it's just the average of sample errors over the batch
        del_biases = [errors.sum(axis=1).reshape(-1,1) / m]
        
        #calculate gradient with respect to weight:
        #multiply error by activations of previous layer
        #then take average across m samples to get weight gradient
        del_weights = [(errors @ activations[-2].T / m)]
        
        #now loop through previous layers and do the same
        for l in range(2, len(self.sizes)):
            errors = (self.weights[-l+1].T @ errors) * \
                    sigmoid(zs[-l], derivative = True)
            del_bias = (errors.sum(axis=1).reshape(-1,1)) / m
            del_biases.append(del_bias)
            del_weight = (errors @ activations[-l-1].T) / m
            del_weights.append(del_weight)
        
        return del_weights, del_biases 

    def train(self, training_data, epochs, learning_rate,
            batch_size = 0, test_data = None, evaluate_per=100):
        """
        Use batched gradient descent on training data over a number of 
        cycles to tune weights and parameters.

        Args:
            training_data (tuple of numpy arrays): an X,Y pair of 
                input features and (one-hot encoded) output labels
            epochs (int): number of cycles over which to run gradient descent
            learning_rate (float): small positive multiplier to prevent
                gradient descent from overshooting when updating parameters
            batch_size (int): if 0, call gradient descent over entire 
                training set, otherwise determines size of minibatches.
            test_data (tuple of numpy arrays): Defaults to None. If provided, 
                evaluate performance of model on test_data. Useful to 
                avoid overfitting.
            evaluate_per (int): Number of epochs between evaluations:
                during an evaluation, mean square error and accuracy is 
                caluclated for the test dataset.

        """
        #training_data is a tuple (X,Y) of inputs and observations
        if training_data[0].shape[0] != training_data[1].shape[0]:
            raise TrainingError("X and Y have different sample sizes!")
        elif training_data[0].shape[1] != self.sizes[0]:
            raise TrainingError("{} feature(s) in inputs, expected {}"\
                    .format(training_data[0].shape[1], self.sizes[0]))
        elif training_data[1].shape[1] != self.sizes[-1]:
            raise TrainingError("{} class(es) in outputs, expected{}"\
                    .format(training_data[1].shape[1], self.sizes[-1]))

        #number observations in sample
        n = training_data[0].shape[0]
        if batch_size == 0:
            batch_size = n
        for i in range(epochs):
            np.random.seed(3 + i) 
            #shuffle the index
            idx = np.random.permutation(n)
            X = training_data[0][idx].copy()
            Y = training_data[1][idx].copy() 
            for j in range(0, n, batch_size):
                X_batch = X[j: j + batch_size]
                Y_batch = Y[j: j + batch_size]
                self.update_batch(X_batch, Y_batch, learning_rate)
            if (test_data is not None) and not ((i+1) % evaluate_per):
                mse, n_correct = self.evaluate(test_data)
                n_test = test_data[0].shape[0]
                print("Evaluating Epoch {}".format(i+1))
                print("--> MSE: {:.2f}".format(mse))
                print("--> Correct prediction: {} / {}".format(n_correct, n_test))
                print("----> {:.2f}%".format(100 * n_correct / n_test))

    def evaluate(self, test_data):
        """
        Evaluate performance of model on test data.
        
        Args:
            test_data (tuple of numpy arrays): X,Y corresponding to input and
            output of test dataset.

        Returns:
            mse (float): Mean square error between model's prediction (Y_hat) 
                and true labels (Y)
            n_correct (int): Number of correct guesses based on highest
                probability assigned to a class (multi-class classification)
                or a threshold of 0.5 (binary classification).
        """
        X,Y = test_data
        n = X.shape[0]
        Y_hat = self.feed_forward(X.T).T
        mse = np.sum(np.linalg.norm(quadratic_cost(Y_hat, Y), axis=1)) / n
        #mse = np.sum(quadratic_cost(Y_hat, Y)) / n
        prediction = Y_hat.copy()
        labels = Y
        #single class classification:
        if labels.shape[1] == 1:
            prediction[prediction > 0.5] = 1
            prediction[prediction <= 0.5] = 0
        else:
            prediction = np.argmax(Y_hat, axis=1)
            labels = np.argmax(Y, axis=1)    
        n_correct = (prediction==labels).sum()
        return mse, n_correct

