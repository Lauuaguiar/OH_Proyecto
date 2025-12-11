import numpy as np

class Trainer:
    def __init__(self, network, optimizer, loss_fn, patience=5, verbose=True):
        """
        patience: número de epochs sin mejora en validación antes de parar (early stopping)
        """
        self.net = network
        self.opt = optimizer
        self.loss_fn = loss_fn
        self.patience = patience
        self.verbose = verbose

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        n = X_train.shape[0]
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        best_val_loss = np.inf
        best_params = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_train_shuffled = X_train[idx]
            y_train_shuffled = y_train[idx]

            epoch_losses = []
            # mini-batches
            for i in range(0, n, batch_size):
                xb = X_train_shuffled[i:i+batch_size]
                yb = y_train_shuffled[i:i+batch_size]

                out = self.net.forward(xb)
                loss = self.loss_fn.forward(out, yb)
                epoch_losses.append(loss)

                grad = self.loss_fn.backward()
                self.net.backward(grad)
                self.opt.update(self.net.params(), self.net.grads())

            # métricas de validación
            val_pred = self.net.forward(X_val)
            val_loss = self.loss_fn.forward(val_pred, y_val)
            val_acc = np.mean(
                np.argmax(val_pred, axis=1) ==
                np.argmax(y_val, axis=1)
            )

            train_loss_epoch = np.mean(epoch_losses)

            history["train_loss"].append(train_loss_epoch)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs} - TrainLoss={train_loss_epoch:.4f} - ValLoss={val_loss:.4f} - ValAcc={val_acc:.4f}")

            # early stopping: guardar mejores pesos
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                # guardar copia de parámetros (deep copy)
                best_params = [p.copy() for p in self.net.params()]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"Early stopping en epoch {epoch+1}. Mejor ValLoss={best_val_loss:.6f}")
                break

        # restaurar mejores parámetros si tenemos
        if best_params is not None:
            for p, best in zip(self.net.params(), best_params):
                p[:] = best

        return history
