# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:49:00 2021

@author: iverm
"""

from sklearn.datasets import make_moons, make_circles
import numpy as np
import matplotlib.pyplot as plt

# generate data
Xtr, Ytr = make_circles(1000)
Xte, Yte = make_circles(1000)

# convert labels to one-hot encoded vectors
Ytr = np.hstack((Ytr.reshape((1000, 1)), (Ytr == 0).reshape(1000, 1)))
Yte = np.hstack((Yte.reshape((1000, 1)), (Yte == 0).reshape(1000, 1)))

class ActivationFunction:
    '''
    Class for activation functions
    '''
    
    def __init__(self, function):
        '''
        

        Parameters
        ----------
        function : String
            Activation function to be used. Alternatives are
            'relu', 'tanh' and 'sigmoid'.

        Returns
        -------
        None.

        '''
        self.function = function
    
    def __call__(self, x):
        function = self.function
        
        if function == 'relu':
            return x * (x > 0)
        elif function == 'tanh':
            return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        elif function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise NameError('function not defined')
        
    def diff(self, x):
        function = self.function
        
        if function == 'relu':
            return 1 * (x > 0)
        elif function == 'tanh':
            return 1 - self(x)**2
        elif function == 'sigmoid':
            return self(x)*(1 - self(x))
        else:
            raise NameError('function not defined')

def softmax(x):
    '''softmax output activation function'''
    n, p = x.shape
    denom = np.repeat(np.sum(np.exp(x), axis = 1).reshape(n, 1), p, axis = 1)
    
    return np.exp(x) / denom

def cross_entropy(x, y):
    '''cross entropy cost function'''
    return -np.sum(x * np.log(y))

class MLP:
    '''
    Class for a 3 Layer Perceptron with 2 output neurons.
    Output activation function is the softmax function.
    The cost function which is minimized is the cross-entropy function.
    '''
    
    def __init__(self, k1, k2, mu, activation_function):
        '''
        

        Parameters
        ----------
        k1 : Int
            # neurons in 1st hidden layer.
        k2 : Int
            # neurons in 2nd hidden layer.
        mu : Float
            Learning rate for the gradient descent algorithm.
        activation_function : Optional
            Activation function for the hidden layers.

        Returns
        -------
        None.

        '''
        self.k1 = k1
        self.k2 = k2
        self.k3 = 2
        self.mu = mu
        self.phi = activation_function

    def init_weights(self):
        '''
        Initialize weights randomly.

        Returns
        -------
        None.

        '''
        
        self.W1 = np.random.uniform(-0.1, 0.1, (self.p + 1, self.k1))
        self.W2 = np.random.uniform(-0.1, 0.1, (self.k1 + 1, self.k2))
        self.W3 = np.random.uniform(-0.1, 0.1, (self.k2 + 1, self.k3))
        
    def forward_pass(self, X):
        '''
        Computes output lables from X.

        Parameters
        ----------
        X : numpy.ndarray
            n x p matrix of feature vectors.

        Returns
        -------
        None.

        '''
        
        n, p = np.shape(X)
        X = np.hstack((np.ones((1000, 1)), X))

        self.Z1 = X @ self.W1
        self.Y1 = np.hstack((np.ones((n, 1)), self.phi(self.Z1)))

        self.Z2 = self.Y1 @ self.W2
        self.Y2 = np.hstack((np.ones((n, 1)), self.phi(self.Z2)))

        self.Z3 = self.Y2 @ self.W3
        self.Y3 = softmax(self.Z3)

    def backward_pass(self, Y):
        '''
        Computes gradients by backpropagation.

        Parameters
        ----------
        Y : numpy.ndarray
            n x 2 matrix of one-hot encoded label vectors.

        Returns
        -------
        None.

        '''
        n = np.shape(Y)[0]
        
        self.d3 = self.Y3 - Y
        
        self.d2 = np.zeros((n, self.k2))
        
        for i in range(n):
            for j in range(self.k2):
                self.d2[i, j] = np.sum(self.d3[i, :] * self.W3[j, :]) * self.phi.diff(self.Z2[i, j])
        
        self.d1 = np.zeros((n, self.k1 + 1))
        
        for i in range(n):
            for j in range(self.k1):
                self.d1[i, j] = np.sum(self.d2[i, :] * self.W2[j, :]) * self.phi.diff(self.Z1[i, j])
        
    def update_weights(self, X):
        '''
        Update parameter weights.

        Parameters
        ----------
        X : numpy.ndarray
            n x p matrix of feature vectors.

        Returns
        -------
        None.

        '''
        n, p = np.shape(X)
        Y0 = np.hstack((np.ones((n, 1)), X))

        for j in range(1, self.k1):
            self.W1[:,j] -= self.mu * np.sum(np.tile(
                self.d1[:, j, None], (1, p + 1)) * Y0, axis = 0)

        for j in range(self.k2):            
            for k in range(self.k1 + 1):
                self.W2[k, j] -= self.mu * np.sum(self.d2[:, j] * self.Y1[:, k])
                    
        for j in range(self.k3):          
            for k in range(self.k2 + 1):
                self.W3[k, j] -= self.mu * np.sum(self.d3[:, j] * self.Y2[:, k])            
    
    def train(self, X, Y, epochs):
        '''
        Train model in epochs.

        Parameters
        ----------
        X : numpy.ndarray
            n x p matrix of feature vectors.
        Y : numpy.ndarray
            n x 2 matrix of labels.
        Epochs : Int
            # epochs.
        
        Returns
        -------
        None.

        '''
        n, self.p = np.shape(X)
                
        self.init_weights()
        
        self.training_error = np.zeros(epochs)
        
        for i in range(epochs):
            self.forward_pass(X)
            self.backward_pass(Y)
            self.update_weights(X)
            self.training_error[i] = self.error(Y)

    def error(self, Y):
        '''
        Compute entropy while training. 

        Parameters
        ----------
        Y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return -np.sum(Y * np.log(self.Y3))

    def classify(self, X):
        '''
        Computes output lablels from X.

        Parameters
        ----------
        X : numpy.ndarray
            n x p matrix of feature vectors.

        Returns
        -------
        None.

        '''
        
        n, p = np.shape(X)
        X = np.hstack((np.ones((1000, 1)), X))

        Z1 = X @ self.W1
        Y1 = np.hstack((np.ones((n, 1)), self.phi(Z1)))

        Z2 = Y1 @ self.W2
        Y2 = np.hstack((np.ones((n, 1)), self.phi(Z2)))

        Z3 = Y2 @ self.W3
        Y3 = softmax(Z3)
        
        return np.round(Y3)


model = MLP(30, 30, 0.001, ActivationFunction('sigmoid'))

model.train(Xtr, Ytr, 1000)

y_pred = model.classify(Xte)

Ytr_pred = model.classify(Xtr)

plt.plot(model.training_error)

plt.scatter(Xtr[np.where(Ytr_pred[:, 0] == 1), 0], Xtr[np.where(Ytr_pred[:, 0] == 1), 1])
plt.scatter(Xtr[np.where(Ytr_pred[:, 1] == 1), 0], Xtr[np.where(Ytr_pred[:, 1] == 1), 1])

plt.scatter(Xte[np.where(y_pred[:, 0] == 1), 0], Xte[np.where(y_pred[:, 0] == 1), 1])
plt.scatter(Xte[np.where(y_pred[:, 1] == 1), 0], Xte[np.where(y_pred[:, 1] == 1), 1])
