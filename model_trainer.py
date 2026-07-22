"""
ModelTrainer — trains, evaluates, and saves all anomaly detection models.

Run this script directly to train all models on synthetic data and save them:
    python ml_models/anomaly_detection/model_trainer.py

Or import and call programmatically:
    from ml_models.anomaly_detection.model_trainer import ModelTrainer
    trainer = ModelTrainer()
    trainer.train_all()
    report = trainer.evaluate_all()
"""
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml_models.anomaly_detection.isolation_forest  import IsolationForestDetector
from ml_models.anomaly_detection.autoencoder        import DenseAutoencoder
from ml_models.anomaly_detection.lstm_anomaly_detector import LSTMAnomalyDetector
from shared.config import (
    VOLTAGE_MIN, VOLTAGE_MAX, CURRENT_MIN, CURRENT_MAX,
    TEMP_MIN, TEMP_MAX, HARMONIC_MIN, HARMONIC_MAX, LOAD_MIN, LOAD_MAX,
    FAULT_VOLTAGE_LOW, FAULT_TEMP_HIGH, FAULT_HARMONIC_HIGH, FAULT_LOAD_HIGH,
)
from shared.utils import get_logger

logger = get_logger(__name__)


def _make_normal(n: int) -> list[dict]:
    import random
    return [
        {
            "voltage":         random.uniform(VOLTAGE_MIN,  VOLTAGE_MAX),
            "current":         random.uniform(CURRENT_MIN,  CURRENT_MAX),
            "temperature":     random.uniform(TEMP_MIN,     TEMP_MAX),
            "harmonic_5th":    random.uniform(HARMONIC_MIN, HARMONIC_MAX),
            "load_percentage": random.uniform(LOAD_MIN,     LOAD_MAX),
        }
        for _ in range(n)
    ]


def _make_faulty(n: int) -> list[dict]:
    import random
    return [
        {
            "voltage":         random.uniform(FAULT_VOLTAGE_LOW, FAULT_VOLTAGE_LOW + 20),
            "current":         random.uniform(22, 30),
            "temperature":     random.uniform(FAULT_TEMP_HIGH, FAULT_TEMP_HIGH + 30),
            "harmonic_5th":    random.uniform(FAULT_HARMONIC_HIGH, 18),
            "load_percentage": random.uniform(FAULT_LOAD_HIGH, 98),
        }
        for _ in range(n)
    ]


class ModelTrainer:
    def __init__(self):
        self.if_detector   = IsolationForestDetector()
        self.ae_detector   = DenseAutoencoder()
        self.lstm_detector = LSTMAnomalyDetector()

    # ── Train ─────────────────────────────────────────────────────────────────

    def train_all(self, n_normal: int = 800, epochs: int = 200) -> None:
        """Train all three models and save to disk."""
        logger.info("=" * 60)
        logger.info("Training all anomaly detection models…")
        logger.info("=" * 60)

        normal_packets = _make_normal(n_normal)
        X_normal = np.array([
            [p["voltage"], p["current"], p["temperature"], p["harmonic_5th"], p["load_percentage"]]
            for p in normal_packets
        ])

        # IsolationForest
        t0 = time.time()
        self.if_detector.train(X_normal)
        self.if_detector.save()
        logger.info(f"IsolationForest trained in {time.time()-t0:.1f}s")

        # Dense Autoencoder
        t0 = time.time()
        self.ae_detector.train(X_normal, epochs=epochs)
        self.ae_detector.save()
        logger.info(f"DenseAutoencoder trained in {time.time()-t0:.1f}s")

        # LSTM Detector
        t0 = time.time()
        self.lstm_detector.train(epochs=min(epochs, 100))
        self.lstm_detector.save()
        logger.info(f"LSTMDetector trained in {time.time()-t0:.1f}s")

        logger.info("All models trained and saved.")

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate_all(self, n_test: int = 200) -> dict:
        """
        Evaluate all models on normal + faulty test data.
        Returns precision, recall, F1 for each model.
        """
        normal_test = _make_normal(n_test)
        faulty_test = _make_faulty(n_test)
        all_packets = normal_test + faulty_test
        true_labels = [0] * n_test + [1] * n_test  # 0=normal, 1=anomaly

        results = {}
        for name, detector in [
            ("IsolationForest", self.if_detector),
            ("DenseAutoencoder", self.ae_detector),
        ]:
            preds = [1 if detector.predict(p)["anomaly"] else 0 for p in all_packets]
            tp = sum(1 for p, t in zip(preds, true_labels) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(preds, true_labels) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(preds, true_labels) if p == 0 and t == 1)
            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)
            f1        = 2 * precision * recall / (precision + recall + 1e-9)
            results[name] = {
                "precision": round(precision, 3),
                "recall":    round(recall, 3),
                "f1":        round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn,
            }
            logger.info(f"{name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all(n_normal=800, epochs=200)
    print("\n--- Evaluation ---")
    report = trainer.evaluate_all()
    for model, metrics in report.items():
        print(f"{model}: {metrics}")
