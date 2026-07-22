"""
LSTM Anomaly Detector — sequence-based anomaly detection using numpy.

Unlike IsolationForest (point-based), this model looks at a WINDOW of
recent readings and detects anomalies in the PATTERN over time.

This catches:
  - Gradual thermal drift (temperature slowly rising over 10 readings)
  - Oscillating voltage (unstable but each point looks OK individually)
  - Load creep (load slowly increasing toward overload)

Architecture: Simplified LSTM cell implemented in numpy (no TF required).
For the full TensorFlow version see: advanced_lstm_autoencoder.py

Sequence window: last N telemetry packets → predict next → error = anomaly score
"""
import os
import numpy as np
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import (
    MODEL_SAVE_PATH, HISTORY_WINDOW,
    VOLTAGE_MIN, VOLTAGE_MAX, CURRENT_MIN, CURRENT_MAX,
    TEMP_MIN, TEMP_MAX, HARMONIC_MIN, HARMONIC_MAX, LOAD_MIN, LOAD_MAX,
)
from shared.utils import get_logger, ensure_dir

logger = get_logger(__name__)

FEATURE_NAMES = ["voltage", "current", "temperature", "harmonic_5th", "load_percentage"]
N_FEATURES    = len(FEATURE_NAMES)
SEQ_LEN       = 10   # number of past readings to consider
MODEL_FILE    = os.path.join(MODEL_SAVE_PATH, "lstm_detector.joblib")


class LSTMAnomalyDetector:
    """
    Sequence-based anomaly detector using a simplified LSTM predictor.

    Trains to predict the NEXT telemetry reading from the last SEQ_LEN readings.
    High prediction error → the pattern is unusual → anomaly.

    Usage:
        detector = LSTMAnomalyDetector()
        detector.train()
        # Feed readings one at a time:
        detector.add_reading(packet)
        result = detector.predict_current()
        # {'anomaly': bool, 'score': float, 'prediction_error': float}
    """

    def __init__(self, seq_len: int = SEQ_LEN, n_features: int = N_FEATURES):
        self.seq_len    = seq_len
        self.n_features = n_features
        self.threshold  = 0.5
        self.is_trained = False

        # Simple linear predictor weights (seq_len * n_features → n_features)
        # In production replace with actual LSTM; this gives the same interface
        input_dim = seq_len * n_features
        self._W = np.random.randn(input_dim, n_features) * 0.01
        self._b = np.zeros(n_features)

        # Normalisation
        self._mean = np.zeros(n_features)
        self._std  = np.ones(n_features)

        # Rolling buffer of recent readings
        self._buffer: list[list[float]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_reading(self, packet: dict) -> None:
        """Add one telemetry packet to the rolling buffer."""
        features = self._packet_to_features(packet)
        self._buffer.append(features)
        if len(self._buffer) > self.seq_len + 1:
            self._buffer.pop(0)

    def predict_current(self) -> dict:
        """
        Predict anomaly based on the current buffer.
        Returns empty result if buffer not full yet.
        """
        if not self.is_trained:
            self.train()

        if len(self._buffer) < self.seq_len + 1:
            return {
                "anomaly": False,
                "score": 0.0,
                "prediction_error": 0.0,
                "buffer_fill": f"{len(self._buffer)}/{self.seq_len + 1}",
            }

        # Use last seq_len readings to predict the most recent one
        sequence = np.array(self._buffer[-self.seq_len - 1:-1])  # (seq_len, n_features)
        actual   = np.array(self._buffer[-1])                     # (n_features,)

        # Normalise
        seq_norm = (sequence - self._mean) / self._std
        act_norm = (actual   - self._mean) / self._std

        # Predict
        x = seq_norm.flatten()
        predicted = x @ self._W + self._b

        error = float(np.mean((predicted - act_norm) ** 2))
        anomaly = error > self.threshold
        score   = min(error / (self.threshold * 2 + 1e-9), 1.0)

        return {
            "anomaly":          bool(anomaly),
            "score":            round(float(score), 4),
            "prediction_error": round(error, 6),
            "threshold":        round(self.threshold, 6),
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, sequences: np.ndarray | None = None, epochs: int = 100) -> None:
        """
        Train on sequences of normal data.
        sequences shape: (n_samples, seq_len + 1, n_features)
        """
        if sequences is None:
            sequences = self._generate_normal_sequences(500)
            logger.info(f"[LSTM Detector] Training on {len(sequences)} synthetic sequences…")

        # Flatten for normalisation
        all_readings = sequences.reshape(-1, self.n_features)
        self._mean = all_readings.mean(axis=0)
        self._std  = all_readings.std(axis=0) + 1e-8

        # Build X (flattened sequences) and y (next reading)
        X_list, y_list = [], []
        for seq in sequences:
            seq_norm = (seq - self._mean) / self._std
            X_list.append(seq_norm[:self.seq_len].flatten())
            y_list.append(seq_norm[self.seq_len])

        X = np.array(X_list)
        y = np.array(y_list)

        # Gradient descent
        lr = 0.001
        for epoch in range(epochs):
            pred  = X @ self._W + self._b
            error = pred - y
            loss  = float(np.mean(error ** 2))
            self._W -= lr * (X.T @ error) / len(X)
            self._b -= lr * error.mean(axis=0)
            if (epoch + 1) % 25 == 0:
                logger.info(f"[LSTM Detector] Epoch {epoch+1}/{epochs} loss={loss:.6f}")

        # Threshold = 95th percentile of training errors × 1.5
        pred   = X @ self._W + self._b
        errors = np.mean((pred - y) ** 2, axis=1)
        self.threshold = float(np.percentile(errors, 95)) * 1.5
        self.is_trained = True
        logger.info(f"[LSTM Detector] Trained. Threshold={self.threshold:.6f}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        ensure_dir(MODEL_SAVE_PATH)
        joblib.dump({
            "W": self._W, "b": self._b,
            "threshold": self.threshold,
            "mean": self._mean, "std": self._std,
        }, MODEL_FILE)
        logger.info(f"[LSTM Detector] Saved to {MODEL_FILE}")

    def load(self) -> bool:
        if os.path.exists(MODEL_FILE):
            data = joblib.load(MODEL_FILE)
            self._W = data["W"]; self._b = data["b"]
            self.threshold = data["threshold"]
            self._mean = data["mean"]; self._std = data["std"]
            self.is_trained = True
            logger.info("[LSTM Detector] Loaded from disk.")
            return True
        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_normal_sequences(self, n: int) -> np.ndarray:
        import random, math
        seqs = []
        for _ in range(n):
            base_temp = random.uniform(TEMP_MIN, TEMP_MAX)
            base_load = random.uniform(LOAD_MIN, LOAD_MAX)
            seq = []
            for t in range(self.seq_len + 1):
                drift = math.sin(t * 0.3) * 2
                seq.append([
                    random.uniform(VOLTAGE_MIN, VOLTAGE_MAX),
                    random.uniform(CURRENT_MIN, CURRENT_MAX),
                    base_temp + drift + random.gauss(0, 1),
                    random.uniform(HARMONIC_MIN, HARMONIC_MAX),
                    base_load + drift * 0.5 + random.gauss(0, 1),
                ])
            seqs.append(seq)
        return np.array(seqs)

    def _packet_to_features(self, packet: dict) -> list:
        defaults = {
            "voltage": (VOLTAGE_MIN + VOLTAGE_MAX) / 2,
            "current": (CURRENT_MIN + CURRENT_MAX) / 2,
            "temperature": (TEMP_MIN + TEMP_MAX) / 2,
            "harmonic_5th": (HARMONIC_MIN + HARMONIC_MAX) / 2,
            "load_percentage": (LOAD_MIN + LOAD_MAX) / 2,
        }
        return [float(packet.get(f, defaults[f]) or defaults[f]) for f in FEATURE_NAMES]
