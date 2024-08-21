import numpy as np

class Loss:
    def calculate(self, y_true, y_pred):
        pass
    
    def gradient(self, y_true, y_pred):
        pass

class MeanSquaredError(Loss):
    def calculate(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size
