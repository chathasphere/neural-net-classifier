import numpy as np
import pdb

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def dsigmoid(Z):
    sig = sigmoid(Z)
    return sig*(1-sig)

def drelu(Z):
     Z_ = np.copy(Z)
     Z_[Z_ > 0] = 1
     Z_[Z_ <= 0] = 0
     return Z


#this would be like Mean Square Error but it's for a *single* observation

def quadratic_cost(y, y_hat):
    #calculate squared error for a single observation (y) and predictor (y_hat_
    return 0.5 * (y - y_hat)**2

def d_quadratic_cost(y, y_hat):
     return (y_hat - y)

class TrainingError(Exception):
    pass

class NeuralNetwork:
    functions = {'relu': relu, 'sigmoid': sigmoid}
    fprimes = {'relu': drelu, 'sigmoid': dsigmoid} 
    def __init__(self, sizes, activations=None):
        self.sizes = sizes
        self.weights = self.init_weights()
        if activations is None:
            self.activations = ["relu" for i in range(1, len(sizes))] + ["sigmoid"]
        else:
            self.activations = activations
        #no bias needed for first layer
        self.biases = [np.random.randn(s,1) * 0.1 for s in sizes[1:]]
        #self.lr = 0.003

    def init_weights(self):
        input_sizes = self.sizes[:-1]
        output_sizes = self.sizes[1:]
        weight_dims = list(zip(output_sizes, input_sizes))
        #at layer l, weight[j][k] is the weight from the 
        # kth neuron in the layer l-1 to the jth neuron in layer l
        weights = [np.random.randn(j,k) for j,k in weight_dims]
        #normalize variance by size of inputs
        weights = [W / W.shape[1]**(0.5) for W in weights]
        return weights

    def forward_prop_layer(self, layer_number, A):
        #A the "activated" matrix input from previous layer
        W = self.weights[layer_number]
        b = self.biases[layer_number]
        Z = np.dot(W, A) + b
        activate = self.functions[self.activations[layer_number]]
        A = activate(Z)
        return A,Z

    def feed_forward(self, X):
        A = X
        for l in range(0, len(self.sizes) - 1):
            A,Z = self.forward_prop_layer(l,A)
        return A

    def update_batch(self, X, Y, learning_rate):
        #X,Y are mini batches of training X and Y
        m = X.shape[0]
        eta = learning_rate
        #Sum the cost function gradient (w.r.t weights, biases) for each 
        #data point in the mini-batch
        #loop through non-input layers
        # i am skeptical this is gonna work
        del_weights, del_biases = self.backpropagate(X,Y,m)
        for l in range(len(self.sizes) - 1):
            self.weights[l] -= eta * del_weights[-l-1]
            self.biases[l] -= eta * del_biases[-l-1]
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
            zs.append(A)
        #backward pass, beginning at last layer
        #errors is an (n,m) array
        #n number of neurons in the layer, m the number of samples in mini-batch
        errors =  np.multiply(d_quadratic_cost(Y.T, activations[-1]), 
                dsigmoid(zs[-1]))
        #take the average of sample errors to get bias gradient
        del_biases = [errors.sum(axis=1).reshape(-1,1) / m]
        #multiply error by activations of previous layer
        #then take average across m samples to get weight gradient
        del_weights = [(errors @ activations[-2].T / m)]
        for l in range(2, len(self.sizes)):
            errors = np.multiply((self.weights[-l+1].T @ errors), dsigmoid(zs[-l]))
            del_bias = (errors.sum(axis=1).reshape(-1,1)) / m
            del_biases.append(del_bias)
            del_weight = (errors @ activations[-l-1].T) / m
            del_weights.append(del_weight)
        
        return del_weights, del_biases 
        #set 0th activation 

    def train(self, training_data, epochs, batch_size,
            learning_rate, test_data = None):
        #training_data is a tuple (X,Y) of inputs and observations
        X,Y = training_data
        if X.shape[0] != Y.shape[0]:
            raise TrainingError("X and Y don't have matching shapes!")
        #number observations in sample
        n = X.shape[0]
#        #baseline performance:
#        Y_hat = self.feed_forward(X.T)
#        mse = np.sum(quadratic_cost(Y_hat, Y.T)) / n
#        print("Baseline MSE loss: {0:.3f}".format(mse))
        for i in range(epochs):
            np.random.seed(3 + i) 
            np.random.shuffle(X)
            np.random.shuffle(Y)
            for j in range(0, n, batch_size):
                X_batch = X[j: j + batch_size]
                Y_batch = Y[j: j + batch_size]
                self.update_batch(X_batch, Y_batch, learning_rate)
            print("Epoch {} complete.".format(i))
            if test_data is not None:
                mse, n_correct = self.evaluate(test_data)
                n_test = test_data[0].shape[0]
                print("--> MSE: {:.2f}".format(mse))
                print("--> Correct prediction: {} / {}".format(n_correct, n_test))

#            Y_hat = self.feed_forward(X.T)
#            mse = np.sum(quadratic_cost(Y_hat, Y.T)) / n
#            print("Epoch {0} complete. MSE loss: {1:.3f}".format(i, mse))

    def evaluate(self, test_data):
        X,Y = test_data
        n = X.shape[0]
        Y_hat = self.feed_forward(X.T).T
        mse = np.sum(quadratic_cost(Y_hat, Y)) / n
        prediction = np.argmax(Y_hat, axis=1)
        labels = np.argmax(Y, axis=1)
        n_correct = (prediction==labels).sum()
        return mse, n_correct


   # def softmax(self, Z, stable=True):
   #     #equivalent to a multiclass logistic function
   #     if stable:
   #         C = np.max(Z)
   #     else:
   #         C = 0
   #     exp_scores = np.exp(Z - C)
   #     return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

   # def dsoftmax(self, Z):
   #     pass

   # def cross_entropy(self, X, y):
   #     #assumes y is one-hot encoded
   #     pass
                            
# replace this eventually with a general "apply classifier" function?
