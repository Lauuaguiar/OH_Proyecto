import numpy as np

class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []
