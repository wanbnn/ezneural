import numpy as np
from .layers import Dense

class Model:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, loss_function, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer
    
    def forward(self, X):
        self.activations = []
        self.inputs = [X]
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
            self.activations.append(activation)
        return activation
    
    def backward(self, y_true, y_pred):
        loss_grad = self.loss_function.gradient(y_true, y_pred)
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, self.optimizer.learning_rate)
    
    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                X_batch, y_batch = X[start:end], y[start:end]
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.optimizer.update(self.layers)
            if epoch % 10 == 0:
                loss = self.loss_function.calculate(y, self.forward(X))
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)
