import numpy as np

class Dense:
    def __init__(self, in_dim, out_dim, initialization="xavier"):
        if initialization == "xavier":
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, +limit, (in_dim, out_dim))
        elif initialization == "he":
            self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
        else:
            self.W = np.random.randn(in_dim, out_dim) * 0.01

        self.b = np.zeros((1, out_dim))
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]
