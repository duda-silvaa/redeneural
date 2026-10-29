"""Rede neural densa implementada do zero com NumPy.

Inclui as ferramentas estudadas na playlist: ativações, perdas, mini-batch
gradient descent, validação, métricas e persistência em pickle.
"""

import pickle
from pathlib import Path

import numpy as np


class NeuralNetwork:
    """MLP para classificação binária com camadas ocultas configuráveis."""

    def __init__(
        self,
        layer_sizes,
        learning_rate=0.01,
        seed=42,
        hidden_activation="relu",
        l2_lambda=0.0,
    ):
        if len(layer_sizes) < 3:
            raise ValueError("A rede precisa de entrada, camada oculta e saída.")
        if hidden_activation not in {"relu", "tanh"}:
            raise ValueError("hidden_activation deve ser 'relu' ou 'tanh'.")
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.l2_lambda = l2_lambda
        rng = np.random.default_rng(seed)
        # He é adequada para ReLU; Xavier é mais estável para Tanh.
        scale = (
            lambda fan_in: np.sqrt(2 / fan_in)
            if hidden_activation == "relu"
            else np.sqrt(1 / fan_in)
        )
        self.weights = [
            rng.normal(0, scale(layer_sizes[i]), (layer_sizes[i], layer_sizes[i + 1]))
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        values = np.tanh(z)
        return 1 - values**2

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def softplus(z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        eps = 1e-8
        return float(-np.mean(
            y_true * np.log(y_pred + eps)
            + (1 - y_true) * np.log(1 - y_pred + eps)
        ))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        eps = 1e-8
        return float(-np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)))

    @staticmethod
    def mse(y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def huber(y_true, y_pred, delta=1.0):
        error = np.abs(y_true - y_pred)
        quadratic = np.minimum(error, delta)
        linear = error - quadratic
        return float(np.mean(0.5 * quadratic**2 + delta * linear))

    def _hidden_activation(self, z):
        return self.relu(z) if self.hidden_activation == "relu" else self.tanh(z)

    def _hidden_derivative(self, z):
        return (
            self.relu_derivative(z)
            if self.hidden_activation == "relu"
            else self.tanh_derivative(z)
        )

    def forward(self, x):
        activations = [x]
        zs = []
        current = x
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weights + bias
            zs.append(z)
            current = self.sigmoid(z) if index == len(self.weights) - 1 else self._hidden_activation(z)
            activations.append(current)
        return activations[-1], activations, zs

    def backward(self, y_true, activations, zs):
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)
        batch_size = y_true.shape[0]
        delta = activations[-1] - y_true
        for index in reversed(range(len(self.weights))):
            gradients_w[index] = activations[index].T @ delta / batch_size
            if self.l2_lambda:
                gradients_w[index] += self.l2_lambda * self.weights[index]
            gradients_b[index] = np.mean(delta, axis=0, keepdims=True)
            if index > 0:
                delta = (delta @ self.weights[index].T) * self._hidden_derivative(zs[index - 1])
        for index in range(len(self.weights)):
            self.weights[index] -= self.learning_rate * gradients_w[index]
            self.biases[index] -= self.learning_rate * gradients_b[index]

    def predict_proba(self, x):
        return self.forward(x)[0]

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).astype(int)

    @staticmethod
    def accuracy(y_true, y_pred):
        return float(np.mean(y_true.ravel().astype(int) == y_pred.ravel().astype(int)))

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        true, pred = y_true.ravel().astype(int), y_pred.ravel().astype(int)
        return {
            "tn": int(((true == 0) & (pred == 0)).sum()),
            "fp": int(((true == 0) & (pred == 1)).sum()),
            "fn": int(((true == 1) & (pred == 0)).sum()),
            "tp": int(((true == 1) & (pred == 1)).sum()),
        }

    def classification_report(self, y_true, y_pred):
        confusion = self.confusion_matrix(y_true, y_pred)
        tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": confusion,
        }

    def parameter_count(self):
        return int(sum(weight.size + bias.size for weight, bias in zip(self.weights, self.biases)))

    def fit(
        self,
        x,
        y,
        epochs=350,
        batch_size=64,
        validation_data=None,
        verbose=0,
        early_stopping_patience=None,
        min_delta=0.0,
    ):
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []
        n_samples = x.shape[0]
        best_val_loss = np.inf
        patience_count = 0
        best_weights = None
        best_biases = None
        for epoch in range(epochs):
            order = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = order[start:start + batch_size]
                prediction, activations, zs = self.forward(x[batch])
                self.backward(y[batch], activations, zs)
            prediction, _, _ = self.forward(x)
            train_loss = self.binary_cross_entropy(y, prediction)
            self.loss_history.append(train_loss)
            self.accuracy_history.append(self.accuracy(y, (prediction >= 0.5).astype(int)))
            if validation_data is not None:
                x_val, y_val = validation_data
                val_prediction = self.predict_proba(x_val)
                val_loss = self.binary_cross_entropy(y_val, val_prediction)
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(
                    self.accuracy(y_val, (val_prediction >= 0.5).astype(int))
                )
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_count = 0
                    best_weights = [weight.copy() for weight in self.weights]
                    best_biases = [bias.copy() for bias in self.biases]
                else:
                    patience_count += 1
                if early_stopping_patience and patience_count >= early_stopping_patience:
                    if best_weights is not None:
                        self.weights, self.biases = best_weights, best_biases
                    break
            if verbose and (epoch == 0 or (epoch + 1) % verbose == 0):
                message = f"Época {epoch + 1}/{epochs} | loss={train_loss:.4f}"
                if validation_data is not None:
                    message += f" | val_loss={self.val_loss_history[-1]:.4f}"
                print(message)
        return self

    def save(self, path):
        with Path(path).open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, NeuralNetwork):
            raise TypeError("O arquivo não contém uma NeuralNetwork.")
        return model
