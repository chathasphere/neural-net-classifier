import numpy as np
from .helpers import sigmoid, relu
from . import crossentropycost as centropy
import pdb

class TrainingError(Exception):
    pass

class NeuralNetwork:
    activation_functions = {'relu': relu, 'sigmoid': sigmoid}

    def __init__(self, sizes, activations=None, cost=centropy):
        self.sizes = sizes
        self.weights = self.initialize_weights()
        if activations is None:
            self.activations = ["relu" for i in range(1, len(sizes))] + ["sigmoid"]
        else:
            self.activations = activations
        #no bias needed for first layer
        self.biases = [np.random.randn(s,1) * 0.1 for s in sizes[1:]]
        self.cost = cost

    def initialize_weights(self):
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
        #A the "activated" matrix input from previous layer
        W = self.weights[layer_number]
        b = self.biases[layer_number]
        Z = W @ A + b
        act_fn = self.activations[layer_number]
        activate = NeuralNetwork.activation_functions[act_fn]
        A = activate(Z)
        return A,Z

    def feed_forward(self, X):
        A = X
        for l in range(0, len(self.sizes) - 1):
            A,Z = self.forward_prop_layer(l,A)
        return A

    def update_batch(self, X, Y, learning_rate, regularization):
        #X,Y are mini batches of training X and Y
        m = X.shape[0]
        eta = learning_rate
        lmbda = regularization
        #Sum the cost function gradient (w.r.t weights, biases) for each 
        #data point in the mini-batch
        del_weights, del_biases = self.backpropagate(X,Y,m)
        for l in range(len(self.sizes) - 1):
            #regularize weights
            self.weights[l] = (1 - eta*lmbda) * self.weights[l] - \
                    (eta * del_weights[-l-1])
            self.biases[l] = self.biases[l] - (eta * del_biases[-l-1])
        return

    def backpropagate(self, X, Y, m):
        del_weights = []
        del_biases = []
        #weighted inputs into neuron
        zs = []
        #activated outputs from neuron 
        #set activation of 0th layer: it's the input!
        A = X.T
        activations = [A]
        #feed forward and store As, Zs
        for l in range(0, len(self.sizes) - 1):
            A,Z = self.forward_prop_layer(l, A)
            activations.append(A)
            zs.append(Z)

        #backward pass
        #errors is an (n,m) array for each layer
        #n number of neurons in the layer, m the number of samples in mini-batch
        #to do: refactor this one

        errors = self.cost.errors(Y = Y.T, Y_hat = activations[-1], Z = zs[-1])

        #take the average of sample errors to get bias gradient
        del_biases = [errors.sum(axis=1).reshape(-1,1) / m]

        #multiply error by activations of previous layer
        #then take average across m samples to get weight gradient
        del_weights = [(errors @ activations[-2].T / m)]
        for l in range(2, len(self.sizes)):
            act_fn = self.activations[-l]
            activate = NeuralNetwork.activation_functions[act_fn]
            errors = (self.weights[-l+1].T @ errors) * \
                    activate(zs[-l], derivative = True)
            del_bias = (errors.sum(axis=1).reshape(-1,1)) / m
            del_biases.append(del_bias)
            del_weight = (errors @ activations[-l-1].T) / m
            del_weights.append(del_weight)
        
        return del_weights, del_biases

    def check_shapes(self, training_data):
        #sanity checks to make sure input matrices are of the right shape
        if training_data[0].shape[0] != training_data[1].shape[0]:
            raise TrainingError("X and Y have different sample sizes!")
        elif training_data[0].shape[1] != self.sizes[0]:
            raise TrainingError("{} feature(s) in inputs, expected {}"\
                    .format(training_data[0].shape[1], self.sizes[0]))
        elif training_data[1].shape[1] != self.sizes[-1]:
            raise TrainingError("{} class(es) in outputs, expected{}"\
                    .format(training_data[1].shape[1], self.sizes[-1]))


    def train(self, training_data, epochs, learning_rate, regularization,
            batch_size = -1, evaluation_data = None, epochs_per_print = 10,
            monitor_training = (True, False), monitor_evaluation = (False, True)):
        #training_data is a tuple (X,Y) of inputs and observations
        self.check_shapes(training_data)
        training_loss, training_accuracy = [], []
        evaluation_loss, evaluation_accuracy = [], []

        if monitor_evaluation[0] or monitor_evaluation[1]:
            if evaluation_data is None:
                raise TrainingError("Evaluation data expected but none found")
        #number observations in sample
        n = training_data[0].shape[0]
        if batch_size < 0:
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
                self.update_batch(X_batch, Y_batch, learning_rate, regularization)
            verbose = False
            if not((i+1) % epochs_per_print):
                print("Epoch {} complete".format(i+1))
                verbose = True
            #measure performance on training data
            a,b = self.evaluate(data = (X,Y), training=True, lmbda = regularization, 
                    get_loss = monitor_training[0], get_accuracy = monitor_training[1],
                    verbose = verbose)
            training_loss.extend(a)
            training_accuracy.extend(b) 
            #measure performance on evaluation data
            c,d = self.evaluate(data = evaluation_data, training=False,
                    lmbda = regularization, get_loss = monitor_evaluation[0], 
                    get_accuracy = monitor_evaluation[1], verbose = verbose)
            evaluation_loss.extend(c)
            evaluation_accuracy.extend(d)
        return  {"training_loss": training_loss,
                "training_accuracy": training_accuracy, 
                "evaluation_loss": evaluation_loss,
                "evaluation_accuracy": evaluation_accuracy}


    def evaluate(self, data, training, get_loss, get_accuracy, verbose, lmbda):
        loss, accuracy = [], []
        if not (get_loss or get_accuracy):
            #optimized for speed
            return loss, accuracy
        dataset = "training" if training else "evaluation"
        X,Y = data
        n = X.shape[0]
        Y_hat = self.feed_forward(X.T).T
        if get_loss:
            c0 = self.cost.loss(Y, Y_hat)
            c1 = 0.5 * lmbda * \
                     sum([np.square(w).sum() for w in self.weights]) / n
            loss = [c0 + c1]
            if verbose:
                print("Loss on {} data: {:.2f}".format(dataset, loss[0]))
        if get_accuracy:
            prediction = Y_hat.copy()
            labels = Y
            #single class classification:
            if labels.shape[1] == 1:
                prediction[prediction < 0.5] = 1
                prediction[prediction <= 0.5] = 0
            #multiple class classification
            else:
                prediction = np.argmax(Y_hat, axis=1)
                labels = np.argmax(Y, axis=1)
            n_correct = (prediction==labels).sum()
            accuracy = [100 * n_correct / n]
            if verbose:
                print("Accuracy on {} data: {}/{}".format(dataset, n_correct,n))
                print("--> {:.2f}%".format(accuracy[0]))
        return loss, accuracy




