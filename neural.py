import pickle
from time import localtime, strftime
import os

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layer_dims = [16, 16], model_path = None):
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_layer_dims = hidden_layer_dims

        self._best_accuracy = 0.0
        self._best_params = []

        if model_path != None:
            self._best_accuracy, self._best_params = self.load_params(model_path)

    def initialize_params(self, layer_dims, seed = 13232):
        # Assign random weights and initialize biases
        np.random.seed(seed)
        n = len(layer_dims)

        # Initializes weight and bias parameters
        params = {}

        for layer in range(1, n):
            params['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer-1]) * 0.01
            params['b' + str(layer)] = np.zeros((layer_dims[layer], 1))

        return params

    # Feed the given input through the layers of weights and biases to calculate the output with the sigmoid function.
    def feedforward(self, input, params):
        A_val = input
        caches = []
        # the amount of passes is the amount of layers, note each layer has a bias and a weight thus divide by 2 
        # and subtract one since we use a different function for the last pass
        n = len(params) // 2

        caches.append((None, A_val))

        for i in range(1, n):
            weights = params['W' + str(i)]
            bias = params['b' + str(i)]

            Z_val = weights.dot(caches[-1][1]) + bias
            A_val = self.relu(Z_val)

            caches.append((Z_val, A_val))
        
        # For the last layer use softmax instead of relu
        Z_val_n = params['W' + str(n)].dot(caches[-1][0]) + params['b' + str(n)]
        A_val_n = self.softmax(Z_val_n)

        caches.append((Z_val_n, A_val_n))
        return caches

    def back_propagation(self, Y, params, caches):
        grads = {}

        m = Y.size
        n = len(params) // 2
        
        one_hot_Y = self.one_hot_enc(Y)
        grads['dZ' + str(n)] = caches[n][1] - one_hot_Y
        grads['dW' + str(n)] = 1/m * grads['dZ' + str(n)].dot(caches[n-1][1].T)
        grads['dB' + str(n)] = 1/m * np.sum(grads['dZ' + str(n)])

        for i in range(n - 1, 0, -1):
            grads['dZ' + str(i)] = params['W' + str(i+1)].T.dot(grads['dZ' + str(i + 1)]) * self.drelu(caches[i][0])
            grads['dW' + str(i)] = 1/m * grads['dZ' + str(i)].dot(caches[i-1][1].T)
            grads['dB' + str(i)] = 1/m * np.sum(grads['dZ' + str(i)])

        return grads
    
    def update_params(self, learning_rate, params, grads):
        n = len(params) // 2
        
        for i in range(1, n):
            params['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
            params['b' + str(i)] -= learning_rate * grads['dB' + str(i)]
        
        return params
    
    # Starts to train the parameters
    # this method uses the gradient descent to find the best params by minimizing the cost function.
    def train(self, X, Y, epochs, learning_rate, save=False, output="output/"):
        layer_dims = [self._input_size]

        # if there are no defined hidden layers, make sure there is a layer with dimension
        # of output size so that 2 layers are formed with dimensions
        # input_size x output_size and output_size x output_size
        if len(self._hidden_layer_dims) > 0:
            for k in self._hidden_layer_dims:
                layer_dims.append(k)
        else:
            layer_dims.append(self._output_size)
        
        layer_dims.append(self._output_size)

        params = self.initialize_params(layer_dims)
        for i in range(epochs):
            caches = self.feedforward(X, params)
            grads = self.back_propagation(Y, params, caches)
            params = self.update_params(learning_rate, params, grads)
            accuracy = self.get_accuracy(caches[-1][1], Y)

            if i % 10 == 0:
                print(f"Epoch: {i}")
                print(f"Accuracy: {accuracy}")

            if accuracy > self._best_accuracy:
                self._best_accuracy = accuracy
                self._best_params = params
            
        print(f"Best accuracy reached {self._best_accuracy}")

        if save:
            if not os.path.exists(output):
                os.makedirs(output)
            
            output_file_location = os.path.join(output, "model-" + strftime("%Y-%m-%d-%H-%M-%S", localtime()) + ".pkl")
            with open(output_file_location, 'wb') as file:
                pickle.dump([self._best_accuracy, self._best_params], file)
    
    def predict(self, X, params = None):
        caches = None
        prediction = None

        if params != None:
            caches = self.feedforward(X, params)
        else:
            caches = self.feedforward(X, self._best_params)
        
        prediction = np.argmax(caches[-1][1], 0)
        return prediction
    
    def load_params(self, model_path):
        params, accuracy = None, None
        with open(model_path, 'rb') as file:
            params, accuracy = pickle.load(file)
        
        if params == None:
            print(f"Failure to load parameters from pah: {model_path}")
            return
        
        return params, accuracy

    # Does one hot encoding for the output vector Y
    def one_hot_enc(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T
    
    # A rectified linear unit. Preferable over the sigmoid function for training since it's computationally more efficient.
    def relu(self, z):
        return np.maximum(0, z)
    
    # The derivative of the rectified linear unit.
    def drelu(self, z):
        return z > 0

    # Sigmoid function. Used in forward propagation to calculate change in output
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # Softmax function. Used to determine the probability for the individualistic output vectors values, that is to say, normalizes output to range ]0,1[
    # where each values of the output vector represents the probability that the value is the best one
    def softmax(self, z):
        return np.exp(z) / sum(np.exp(z))
    
    # Stable softmax that avoids overflowing
    def stable_softmax(self, z):
        z = z - np.max(z)
        return self.softmax(z)
    
    def get_accuracy(self, predictions, Y):
        return np.sum(np.argmax(predictions, 0) == Y) / Y.size
    
    def get_best_accuracy(self):
        return self._best_accuracy
    