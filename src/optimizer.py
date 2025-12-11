import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        """
        weight_decay: lambda para L2 (se aplica solo a parámetros con ndim>1, i.e. pesos)
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
        self.weight_decay = weight_decay

    def update(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            # Aplicar weight decay directamente al gradiente si corresponde (pesos)
            if self.weight_decay and p.ndim > 1:
                g = g + self.weight_decay * p

            if i not in self.m:
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, params, grads):
        for p, g in zip(params, grads):
            if self.weight_decay and p.ndim > 1:
                g = g + self.weight_decay * p
            p -= self.lr * g
