# ---------- Neurona Multiclase con Softmax ----------
import numpy as np

class NeuronaSoftmax:
    def __init__(self, alpha=0.1, n_iter=1000, cotaE=1e-5):
        self.alpha = alpha
        self.n_iter = n_iter
        self.cotaE = cotaE

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # estabilidad numérica
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        n_classes = Y.shape[1]

        # Inicialización de pesos
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))

        for _ in range(self.n_iter):
            # Forward
            Z = X @ self.W + self.b
            Y_hat = self.softmax(Z)

            # Costo (categorical cross entropy)
            loss = -np.mean(np.sum(Y * np.log(Y_hat + 1e-15), axis=1))

            # Gradientes
            dW = (X.T @ (Y_hat - Y)) / n_samples
            db = np.mean(Y_hat - Y, axis=0, keepdims=True)

            # Actualización
            self.W -= self.alpha * dW
            self.b -= self.alpha * db

            # Criterio de corte
            if loss < self.cotaE:
                break

    def predict_proba(self, X):
        return self.softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)