import numpy as np

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        # estabilidad numérica: clip
        clipped = np.clip(y_pred, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(clipped), axis=1))

    def backward(self):
        # derivada simplificada para softmax + cross-entropy
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true)**2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
