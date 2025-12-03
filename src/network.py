class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        ps = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                ps.extend(layer.params())
        return ps

    def grads(self):
        gs = []
        for layer in self.layers:
            if hasattr(layer, "grads"):
                gs.extend(layer.grads())
        return gs
