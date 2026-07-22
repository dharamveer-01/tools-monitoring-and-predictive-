"""
Dense Autoencoder anomaly detector (numpy-only, no TensorFlow required).

How it works:
  1. Train on normal telemetry → learns to reconstruct healthy patterns
  2. At inference: reconstruct the input → measure reconstruction error
  3. High error = the pattern is unfamiliar = anomaly

This is lighter than the LSTM version and works without GPU/TF.
Use this when TensorFlow is not installed.

Architecture:
  Input(5) → Dense(16) → ReLU → Dense(8) → ReLU
           → Dense(16) → ReLU → Dense(5)   [reconstruction]
"""
import os
import numpy as np
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import MODEL_SAVE_PATH, VOLTAGE_MIN, VOLTAGE_MAX, CURRENT_MIN, CURRENT_MAX
from shared.config import TEMP_MIN, TEMP_MAX, HARMONIC_MIN, HARMONIC_MAX, LOAD_MIN, LOAD_MAX
from shared.utils import get_logger, ensure_dir

logger = get_logger(__name__)

FEATURE_NAMES = ["voltage", "current", "temperature", "harmonic_5th", "load_percentage"]
MODEL_FILE    = os.path.join(MODEL_SAVE_PATH, "autoencoder_weights.joblib")


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class DenseAutoencoder:
    """
    Lightweight numpy autoencoder for anomaly detection.
    No external deep learning framework required.
    """

    def __init__(self, input_dim: int = 5, hidden_dims: list = None, lr: float = 0.01):
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims or [16, 8, 16]
        self.lr          = lr
        self.threshold   = 0.1
        self.is_trained  = False

        # Build weight matrices
        dims = [input_dim] + self.hidden_dims + [input_dim]
        self.weights = []
        self.biases  = []
        for i in range(len(dims) - 1):
            # Xavier initialisation
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self.weights.append(np.random.randn(dims[i], dims[i + 1]) * scale)
            self.biases.append(np.zeros(dims[i + 1]))

        # Normalisation params (set during training)
        self._mean = np.zeros(input_dim)
        self._std  = np.ones(input_dim)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """X shape: (n, input_dim). Returns reconstruction."""
        h = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = _relu(h)
        return h  # linear output layer

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Returns MSE per sample."""
        recon = self._forward(X)
        return np.mean((X - recon) ** 2, axis=1)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray | None = None, epochs: int = 200) -> None:
        """Train on normal data. X shape: (n_samples, 5)."""
        if X is None:
            X = self._generate_normal_data(600)
            logger.info(f"[Autoencoder] Training on {len(X)} synthetic samples…")

        # Normalise
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        # Mini-batch SGD
        batch_size = 32
        n = len(X_norm)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                batch = X_norm[idx[start:start + batch_size]]
                loss  = self._train_step(batch)
                total_loss += loss
            if (epoch + 1) % 50 == 0:
                logger.info(f"[Autoencoder] Epoch {epoch+1}/{epochs} loss={total_loss/n:.6f}")

        # Set threshold = 95th percentile of training errors × 1.5
        errors = self._reconstruction_error(X_norm)
        self.threshold = float(np.percentile(errors, 95)) * 1.5
        self.is_trained = True
        logger.info(f"[Autoencoder] Trained. Threshold={self.threshold:.6f}")

    def _train_step(self, batch: np.ndarray) -> float:
        """One gradient descent step via backpropagation."""
        # Forward
        activations = [batch]
        h = batch
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = _relu(h)
            activations.append(h)

        recon = activations[-1]
        loss  = float(np.mean((batch - recon) ** 2))

        # Backward
        delta = -2 * (batch - recon) / len(batch)
        for i in reversed(range(len(self.weights))):
            dW = activations[i].T @ delta
            db = delta.sum(axis=0)
            self.weights[i] -= self.lr * dW
            self.biases[i]  -= self.lr * db
            if i > 0:
                delta = delta @ self.weights[i].T
                delta[activations[i] <= 0] = 0  # ReLU gradient

        return loss

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, packet: dict) -> dict:
        if not self.is_trained:
            self.train()

        features = self._packet_to_features(packet)
        X = (np.array([features]) - self._mean) / self._std
        error = float(self._reconstruction_error(X)[0])
        anomaly = error > self.threshold
        score = min(error / (self.threshold * 2 + 1e-9), 1.0)

        return {
            "anomaly":   bool(anomaly),
            "score":     round(float(score), 4),
            "recon_error": round(error, 6),
            "threshold": round(self.threshold, 6),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        ensure_dir(MODEL_SAVE_PATH)
        joblib.dump({
            "weights":   self.weights,
            "biases":    self.biases,
            "threshold": self.threshold,
            "mean":      self._mean,
            "std":       self._std,
        }, MODEL_FILE)
        logger.info(f"[Autoencoder] Saved to {MODEL_FILE}")

    def load(self) -> bool:
        if os.path.exists(MODEL_FILE):
            data = joblib.load(MODEL_FILE)
            self.weights   = data["weights"]
            self.biases    = data["biases"]
            self.threshold = data["threshold"]
            self._mean     = data["mean"]
            self._std      = data["std"]
            self.is_trained = True
            logger.info("[Autoencoder] Loaded from disk.")
            return True
        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_normal_data(self, n: int) -> np.ndarray:
        import random
        return np.array([
            [
                random.uniform(VOLTAGE_MIN,  VOLTAGE_MAX),
                random.uniform(CURRENT_MIN,  CURRENT_MAX),
                random.uniform(TEMP_MIN,     TEMP_MAX),
                random.uniform(HARMONIC_MIN, HARMONIC_MAX),
                random.uniform(LOAD_MIN,     LOAD_MAX),
            ]
            for _ in range(n)
        ])

    def _packet_to_features(self, packet: dict) -> list:
        defaults = {
            "voltage": (VOLTAGE_MIN + VOLTAGE_MAX) / 2,
            "current": (CURRENT_MIN + CURRENT_MAX) / 2,
            "temperature": (TEMP_MIN + TEMP_MAX) / 2,
            "harmonic_5th": (HARMONIC_MIN + HARMONIC_MAX) / 2,
            "load_percentage": (LOAD_MIN + LOAD_MAX) / 2,
        }
        return [float(packet.get(f, defaults[f]) or defaults[f]) for f in FEATURE_NAMES]
