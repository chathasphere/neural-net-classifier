import numpy as np

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
        output_sizes = self.sizes[:1]
        weight_dims = zip(output_sizes, input_sizes)
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

    def train(self, X, y):
        pass

   # def sigmoid(self, Z):
   #     return 1/(1 + np.exp(-Z))

   # def relu(self, Z):
   #     return np.maximum(0,Z)

   # def dsigmoid(self, Z):
   #     sig = sigmoid(Z)
   #     return sig*(1-sig)

   # def drelu(self, Z):
   #     Z_ = np.copy(Z)
   #     Z_[Z_ > 0] = 1
   #     Z_[Z_ <= 0] = 0
   #     return Z

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
O#A

                            

