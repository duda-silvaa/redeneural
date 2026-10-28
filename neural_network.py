"""Rede neural densa implementada do zero com NumPy."""

import numpy as np


class NeuralNetwork:
    """MLP para classificação binária com ReLU nas camadas ocultas."""

    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        if len(layer_sizes) < 3:
            raise ValueError("A rede precisa de entrada, camada oculta e saída.")
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0, np.sqrt(2 / layer_sizes[i]), (layer_sizes[i], layer_sizes[i + 1]))
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]
        self.loss_history = []

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        eps = 1e-8
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def forward(self, x):
        activations = [x]
        zs = []
        current = x
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weights + bias
            zs.append(z)
            current = self.sigmoid(z) if index == len(self.weights) - 1 else self.relu(z)
            activations.append(current)
        return activations[-1], activations, zs

    def backward(self, y_true, activations, zs):
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)
        batch_size = y_true.shape[0]
        delta = activations[-1] - y_true
        for index in reversed(range(len(self.weights))):
            gradients_w[index] = activations[index].T @ delta / batch_size
            gradients_b[index] = np.mean(delta, axis=0, keepdims=True)
            if index > 0:
                delta = (delta @ self.weights[index].T) * self.relu_derivative(zs[index - 1])
        for index in range(len(self.weights)):
            self.weights[index] -= self.learning_rate * gradients_w[index]
            self.biases[index] -= self.learning_rate * gradients_b[index]

    def fit(self, x, y, epochs=350, batch_size=64):
        self.loss_history = []
        n_samples = x.shape[0]
        for _ in range(epochs):
            order = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = order[start:start + batch_size]
                prediction, activations, zs = self.forward(x[batch])
                self.backward(y[batch], activations, zs)
            prediction, _, _ = self.forward(x)
            self.loss_history.append(float(self.binary_cross_entropy(y, prediction)))
        return self

    def predict_proba(self, x):
        return self.forward(x)[0]

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).astype(int)
