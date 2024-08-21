import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.activation_func = self._get_activation_func(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
    
    def _get_activation_func(self, activation):
        if activation == 'relu':
            return self._relu
        elif activation == 'sigmoid':
            return self._sigmoid
        elif activation == 'tanh':
            return self._tanh
        else:
            return None
    
    def _get_activation_derivative(self, activation):
        if activation == 'relu':
            return self._relu_derivative
        elif activation == 'sigmoid':
            return self._sigmoid_derivative
        elif activation == 'tanh':
            return self._tanh_derivative
        else:
            return None
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2
    
    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.bias
        if self.activation_func:
            self.output = self.activation_func(self.output)
        return self.output
    
    def backward(self, dL_dout, learning_rate):
        if self.activation_derivative:
            dL_dout *= self.activation_derivative(self.output)
        dL_dW = np.dot(self.input.T, dL_dout)
        dL_db = np.sum(dL_dout, axis=0, keepdims=True)
        dL_dX = np.dot(dL_dout, self.weights.T)
        self.weights -= learning_rate * dL_dW
        self.bias -= learning_rate * dL_db
        return dL_dX
