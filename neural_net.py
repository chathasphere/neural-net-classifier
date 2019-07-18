import numpy as np
from random import shuffle

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

class NeuralNetwork:
    functions = {'relu': relu, 'sigmoid': sigmoid}
    fprimes = {'relu': drelu, 'sigmoid': dsigmoid} 
    def __init__(self, sizes, activations=None):
        self.sizes = sizes
        self.cache = {}
        self.weights = self.init_weights()
        if activations is None:
            activations = ["relu" for i in range(1, len(sizes))] + ["sigmoid"]
        else:
            self.activations = activations
        #no bias needed for first layer
        self.biases = [np.random.randn(s,1) * 0.1 for s in sizes[1:]]
        #self.lr = 0.003

    def init_weights(self):
        input_sizes = self.sizes[1:]
        output_sizes = self.sizes[:-1]
        weight_dims = list(zip(output_sizes, input_sizes))
        weights = [np.random.randn(r,c) for r,c in weight_dims]
        #normalize variance by size of inputs
        weights = [W / W.shape[1]**(0.5) for W in weights]
        return weights

    def forward_prop_layer(self, layer_number, A):
        #A the "activated" matrix input from previous layer
        W = self.weights[layer_number]
        b = self.biases[layer_number]
        Z = np.dot(W, A) + b
        activation_fn = functions[activations[layer_number]]
        A = activation(Z)
        return A,Z

    def feed_forward(self, X):
        A = X
        for i in range(len(self.layers)):
            A, Z = self.forward_prop_layer(i, A)
	    #cache outputs for backprop
            self.cache['A_{}'.format(i)] = A
            self.cache['Z_{}'.format(i)] = Z
        return A

    def update_batch(self, mini_batch, learning_rate):
        m = len(mini_batch)
        eta = learning_rate
        #Sum the cost function gradient (w.r.t weights, biases) for each 
        #data point in the mini-batch
        sum_del_weights = [np.zeros(W.shape) for W in self.weights]
        sum_del_biases = [np.zeros(b.shape) for b in self.biases]
        #FOOL we're going to do a ~vectorized~ approach 
        #bwahahahahahahahaha
        print(mini_batch[0].shape, mini_batch[1].shape)

#        for x_j, y_j in mini_batch:
#            del_weights, del_biases = self.backpropagate(x_j,y_j)
#            sum_del_weights += del_weights
#            sum_del_biases += del_biases
#
#        for k in range(len(self.sizes) - 1):
#            del_weights
#            self.weights[k] -= (eta/m) * sum_del_weights[k]
#            self.biases[k] -= (eta/m) * sum_del_biases[k]
#            print("{}".format(k))
    def train(self, training_data, epochs, batch_size, learning_rate=0.03):
        #training data list of pairs (x,y) where x is a training input, y the target output
        for i in range(epochs):
            shuffle(training_data)
            n = len(training_data)
            for j in range(0, n, batch_size):
                mini_batch = training_data[j: j + batch_size]
                self.update_batch(mini_batch, learning_rate)
                break       
                print("batch head: {}".format(j))
            #ToDo:
            #Print update on Epochs
            #i guess we could calculate 

    def backpropagate(self, x, y):
        del_weight = [np.zeros(W.shape) for W in self.weights]
        del_bias = [np.zeros(b.shape) for b in self.biases]
        #set 0th activation 
        activated = x

        return del_weight, del_bias



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
                            

