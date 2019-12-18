import numpy as np


class Functions:
    @staticmethod
    def relu(h):
        return np.maximum(0, h)

    @staticmethod
    def softmax(z):
        # Calculate exponent term first
        ex = np.exp(z)
        return ex / np.sum(ex, axis=0, keepdims=True)

    @staticmethod
    def log_loss(y, y_hat):
        return np.sum(-y * np.log(y_hat.clip(min=1e-6)))

    @staticmethod
    def differentiate_relu(x):
        return (x > 0) * 1

    @staticmethod
    def dropout(h, p):
        return np.random.binomial(1, p, size=len(h)) / p