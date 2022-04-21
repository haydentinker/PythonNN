#!/usr/bin/python
'''============================================================================
# Name: Example
# Username: Mikhail Beresnev
# Username: Hayden Tinker
# Course: CPTR330
# Assignment: Lab 4
# Description: Implementation of the ML algorithm Neural Network.
#============================================================================'''
import numpy as np
from scipy import optimize
import scipy

# python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
# may need to restart vs code
from scipy import optimize

class MLAlgorithm:
    '''
    Implement Neural Network
    '''

    def __init__(self, parameters):
        '''
        Initializes all the variable for the algorithm.
        Parameters will hold key:value pairs for defining details in the
        algorithm.
        '''
        self.lambda_value = 0.00001
        self.a_value = list()
        self.z_value = list()

        # Obtain Input Parameters
        self.input_size = int(parameters['input_size']) # nodes
        self.output_size = int(parameters['output_size']) # nodes
        self.hidden_layers = parameters['hidden_layers'].split(",")
        self.hidden_layers = [int(item) for item in self.hidden_layers]

        # Make a list of all layer sizes
        self.layer_sizes = list()
        self.layer_sizes.append(self.input_size)
        for layer in self.hidden_layers:
            self.layer_sizes.append(layer)
        self.layer_sizes.append(self.output_size)

        # Based on the layer sizes, create a list of weights
        self.weights = list()
        input_size = self.input_size
        for layer_size in self.layer_sizes[1:]:
            self.weights.append(np.random.randn(input_size, layer_size))
            input_size = layer_size

    def get_algorithm(self):
        '''
        Returns the name of the algorithm.
        '''
        return "Neural Network"

    def train(self, dataset, labels):
        '''
        Train the model based on the dataset and labels.
        '''
        # Transform data and labels ot np arrays
        dataset = np.array((dataset), dtype = float)
        labels = np.array((labels), dtype = float)

        # Obtain the initial weights
        params0 = self.get_params()

        # Optimize
        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.cost_function_wrapper, params0, jac=True, method='BFGS', \
            args=(dataset, labels), options=options, callback=self.callback_f)

        # Store results
        self.set_params(_res.x)

    def cost_function_wrapper(self, params, data, labels):
        '''
        cost function wrapper for optimize
        '''
        self.set_params(params)
        # Obtain the cost (error) of the current weights
        cost = self.cost_function(data, labels)

        # Find the direction we want to go
        grad = self.compute_gradients(data, labels)

        # cost = single list or array
        # grad = np array
        return cost, grad

    def cost_function(self, data, labels):
        # Calculates the error based on the expected labels

        # Go through the forward function
        y_hat = self.forward(data)
        
        # Calulate the error, mitigating overfitting by regularization
        summations = 0
        for weight in self.weights:
            summations += self.lambda_value/2 * np.sum(weight**2)
        j_value = 0.5*sum((labels-y_hat)**2)/data.shape[0] + summations
        return j_value

    def cost_function_prime(self, data, labels):
        # Backpropogates the error

        # Go through the forward function
        y_hat = self.forward(data)

        # Multiply the error back through the network
        # Use regularization

        dj_dw = list()
        index = len(self.layer_sizes) - 2
        delta = np.multiply(-(labels-y_hat), self.sigmoid_prime(self.z_value[index]))
        dj_dw_temp = np.dot(self.a_value[index-1].T, delta)/data.shape[0] + self.lambda_value*self.weights[index]
        dj_dw.append(dj_dw_temp)

        index = index - 1
        while index >= 0:
            delta = np.dot(delta, self.weights[index+1].T)*self.sigmoid_prime(self.z_value[index])
            if (index != 0): # check to make sure we are not at the last index
                dj_dw_temp = np.dot(self.a_value[index-1].T, delta)/data.shape[0] + self.lambda_value*self.weights[index]
                dj_dw.insert(0, dj_dw_temp)
            index = index - 1
        
        # Multiply the backpropogated error by the original data
        dj_dw_temp = np.dot(data.T, delta)/data.shape[0] + self.lambda_value*self.weights[index-1]
        dj_dw.insert(0, dj_dw_temp)
        return dj_dw

    def forward(self, data):
        # Multiply the data by the weights to get an estimate of the data

        # For backpropogation, we also need to store the weights and activation values
        z_value = np.dot(data, self.weights[0])
        self.z_value.append(z_value) 
        activity = self.sigmoid(z_value)
        for weights in self.weights[1:]:
            self.a_value.append(activity)
            z_value = np.dot(activity, weights)
            self.z_value.append(z_value)
            activity = self.sigmoid(z_value)
        y_hat = activity
        return y_hat

    def compute_numerical_gradient(self, data, labels):
        # Get the direction we want to go, specifically for numeric data

        params_initial = self.get_params()
        numgrad = np.zeros(params_initial.shape)
        perturb = np.zeros(params_initial.shape)
        e_value = 1e-4
        for p_value in range(len(params_initial)):
            #Set perturbation vector
            perturb[p_value] = e_value
            self.set_params(params_initial + perturb)
            loss2 = self.cost_function(data, labels)

            self.set_params(params_initial - perturb)
            loss1 = self.cost_function(data, labels)

            #Compute Numerical Gradient
            vector = (loss2 - loss1) / (2*e_value)
            # Get the magnitude of the vectors to obtain one direction
            magnitude = np.linalg.norm(vector)
            numgrad[p_value] = magnitude

            #Return the value we changed to zero:
            perturb[p_value] = 0

        #Return Params to original value:
        self.set_params(params_initial)
        return numgrad

    def compute_gradients(self, data, labels):
        # Get the direction we want to go during in trying to reduce the cost
        dj_dw = self.cost_function_prime(data, labels)
        dj_dw_ravelled = dj_dw[0]
        for item in dj_dw[1:]:
            dj_dw_ravelled = np.concatenate((np.array(dj_dw_ravelled).ravel(), np.array(item).ravel()))
        return dj_dw_ravelled

    def sigmoid(self, z_value):
        # activation function
        value = scipy.special.expit(z_value)
        return value

    def sigmoid_prime(self,z_value):
        # Derivative of the activation function
        value = self.sigmoid(z_value) * (1 - self.sigmoid(z_value))
        return value

    def callback_f(self, params):
        '''
        call back function for optimizer
        '''
        self.set_params(params)

    def set_params(self, params):
        # Revert the weights back to the original shape based on parameters

        # w_start = w_end
        # w_end = w_start + layer[i] * layer[i+1]
        # reshape
        w_start = 0
        for index in range(len(self.layer_sizes) - 2):
            w_end = w_start + self.layer_sizes[index] * self.layer_sizes[index + 1]
            self.weights[index] = np.reshape(params[w_start:w_end], (self.layer_sizes[index], self.layer_sizes[index+1]))
            w_start = w_end

    def get_params(self):
        # Flatten the weights down to a single row list
        params = np.asarray(self.weights[0]).ravel()
        for weights in self.weights[1:]:
            temp = np.asarray(weights).ravel()
            params = np.concatenate((params, temp))
        return params

    def get_predictions(self, test_set):
        '''
        Return the predictions for testSet using the algorithm model.
        '''

        test_set = np.array((test_set), dtype = float)
        value = self.forward(test_set).tolist()
        return value