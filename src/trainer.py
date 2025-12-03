import numpy as np

class Trainer:
    def __init__(self, network, optimizer, loss_fn):
        self.net = network
        self.opt = optimizer
        self.loss_fn = loss_fn

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        n = X_train.shape[0]
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]

            for i in range(0, n, batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]

                out = self.net.forward(xb)
                loss = self.loss_fn.forward(out, yb)
                grad = self.loss_fn.backward()
                self.net.backward(grad)
                self.opt.update(self.net.params(), self.net.grads())

            val_pred = self.net.forward(X_val)
            val_loss = self.loss_fn.forward(val_pred, y_val)
            val_acc = np.mean(
                np.argmax(val_pred, axis=1) ==
                np.argmax(y_val, axis=1)
            )

            history["train_loss"].append(loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - Loss={loss:.4f} - ValLoss={val_loss:.4f} - ValAcc={val_acc:.4f}")

        return history
