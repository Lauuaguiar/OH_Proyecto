import numpy as np

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true)**2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
