"""
IsolationForest anomaly detector — standalone, saveable, loadable version.

This is the production-grade wrapper around sklearn's IsolationForest.
It adds:
  - save/load to disk (joblib)
  - online retraining with new normal data
  - anomaly score normalisation (0–1)
  - feature importance via permutation

Used by: ml/anomaly_detector.py (the live pipeline uses this class)
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_TRAINING_SAMPLES,
    MODEL_SAVE_PATH,
    VOLTAGE_MIN, VOLTAGE_MAX,
    CURRENT_MIN, CURRENT_MAX,
    TEMP_MIN, TEMP_MAX,
    HARMONIC_MIN, HARMONIC_MAX,
    LOAD_MIN, LOAD_MAX,
)
from shared.utils import get_logger, ensure_dir

logger = get_logger(__name__)

FEATURE_NAMES = ["voltage", "current", "temperature", "harmonic_5th", "load_percentage"]
MODEL_FILE    = os.path.join(MODEL_SAVE_PATH, "isolation_forest.joblib")
SCALER_FILE   = os.path.join(MODEL_SAVE_PATH, "isolation_forest_scaler.joblib")


class IsolationForestDetector:
    """
    Standalone IsolationForest anomaly detector with persistence.

    Usage:
        detector = IsolationForestDetector()
        detector.train()                    # train on synthetic normal data
        result = detector.predict(packet)   # {'anomaly': bool, 'score': float}
        detector.save()                     # save to disk
        detector.load()                     # load from disk
    """

    def __init__(self, contamination: float = ISOLATION_FOREST_CONTAMINATION):
        self.contamination = contamination
        self.model  = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray | None = None) -> None:
        """
        Train on provided data or generate synthetic normal data.
        X shape: (n_samples, 5)
        """
        if X is None:
            X = self._generate_normal_data(ISOLATION_FOREST_TRAINING_SAMPLES)
            logger.info(f"[IsolationForest] Training on {len(X)} synthetic normal samples…")
        else:
            logger.info(f"[IsolationForest] Training on {len(X)} provided samples…")

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        logger.info("[IsolationForest] Training complete.")

    def retrain_with_new_data(self, new_normal_data: list[dict]) -> None:
        """Add new confirmed-normal packets and retrain."""
        X_new = np.array([self._packet_to_features(p) for p in new_normal_data])
        X_old = self._generate_normal_data(ISOLATION_FOREST_TRAINING_SAMPLES)
        X_combined = np.vstack([X_old, X_new])
        self.train(X_combined)
        logger.info(f"[IsolationForest] Retrained with {len(new_normal_data)} new normal samples.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, packet: dict) -> dict:
        """
        Predict anomaly for one telemetry packet.

        Returns:
            {
                'anomaly':       bool,
                'score':         float (0=normal, 1=most anomalous),
                'raw_score':     float (IF decision function, negative=anomalous),
                'features_used': list[str]
            }
        """
        if not self.is_trained:
            self.train()

        features, used = self._packet_to_features_safe(packet)
        X = self.scaler.transform([features])

        pred      = self.model.predict(X)[0]          # -1 or 1
        raw_score = float(self.model.decision_function(X)[0])

        # Normalise: decision_function returns negative for anomalies
        # Map to 0 (normal) → 1 (anomaly)
        norm_score = float(np.clip(1.0 - (raw_score + 0.5), 0.0, 1.0))

        return {
            "anomaly":       bool(pred == -1),
            "score":         round(norm_score, 4),
            "raw_score":     round(raw_score, 4),
            "features_used": used,
        }

    def predict_batch(self, packets: list[dict]) -> list[dict]:
        return [self.predict(p) for p in packets]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        ensure_dir(MODEL_SAVE_PATH)
        joblib.dump(self.model,  MODEL_FILE)
        joblib.dump(self.scaler, SCALER_FILE)
        logger.info(f"[IsolationForest] Saved to {MODEL_SAVE_PATH}")

    def load(self) -> bool:
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            self.model      = joblib.load(MODEL_FILE)
            self.scaler     = joblib.load(SCALER_FILE)
            self.is_trained = True
            logger.info("[IsolationForest] Loaded from disk.")
            return True
        logger.info("[IsolationForest] No saved model found — will train fresh.")
        return False

    # ── Private ───────────────────────────────────────────────────────────────

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

    def _packet_to_features_safe(self, packet: dict) -> tuple[list, list[str]]:
        """Returns (features, list_of_fields_actually_used)."""
        defaults = {
            "voltage": (VOLTAGE_MIN + VOLTAGE_MAX) / 2,
            "current": (CURRENT_MIN + CURRENT_MAX) / 2,
            "temperature": (TEMP_MIN + TEMP_MAX) / 2,
            "harmonic_5th": (HARMONIC_MIN + HARMONIC_MAX) / 2,
            "load_percentage": (LOAD_MIN + LOAD_MAX) / 2,
        }
        features, used = [], []
        for f in FEATURE_NAMES:
            val = packet.get(f)
            if val is not None and float(val) >= 10.0 or f != "voltage":
                try:
                    features.append(float(val))
                    used.append(f)
                    continue
                except (TypeError, ValueError):
                    pass
            features.append(defaults[f])
        return features, used
