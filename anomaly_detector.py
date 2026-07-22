"""
AnomalyDetector — unified anomaly detection for the live pipeline.

Uses the production-grade IsolationForestDetector from ml_models/ which adds:
  - StandardScaler normalisation
  - save/load to disk (persists across restarts)
  - normalised anomaly score (0–1)
  - phone voltage neutralisation

Falls back to a lightweight inline IsolationForest if ml_models/ import fails.
"""
import os
import sys
import numpy as np

# Ensure project root is on path for clean imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.utils import get_logger

logger = get_logger(__name__)

FEATURE_ORDER = ["voltage", "current", "temperature", "harmonic_5th", "load_percentage"]

# Try to use the full ml_models version (with scaler + persistence)
try:
    from ml_models.anomaly_detection.isolation_forest import IsolationForestDetector as _IFDetector
    _USE_FULL = True
    logger.info("[AnomalyDetector] Using ml_models IsolationForestDetector (with scaler + persistence).")
except ImportError as e:
    _USE_FULL = False
    logger.warning(f"[AnomalyDetector] ml_models import failed ({e}), using inline fallback.")


class AnomalyDetector:
    """
    Drop-in anomaly detector for the live pipeline.
    Wraps IsolationForestDetector from ml_models/ when available.
    """

    def __init__(self):
        if _USE_FULL:
            self._detector = _IFDetector()
            # Try to load a previously saved model; train fresh if none exists
            if not self._detector.load():
                self._detector.train()
                self._detector.save()
            self.model = self._detector.model   # expose .model for backward compat
        else:
            self._detector = None
            self._init_fallback()

        self.is_trained = True

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(self, telemetry: dict) -> bool:
        """Returns True if anomaly detected."""
        if _USE_FULL:
            result = self._detector.predict(telemetry)
            return result["anomaly"]
        return self._fallback_detect(telemetry)

    def detect_with_score(self, telemetry: dict) -> dict:
        """Returns full result dict including normalised score."""
        if _USE_FULL:
            return self._detector.predict(telemetry)
        is_anomaly = self._fallback_detect(telemetry)
        return {"anomaly": is_anomaly, "score": 1.0 if is_anomaly else 0.0, "raw_score": 0.0}

    # ── Fallback (inline sklearn, no scaler) ──────────────────────────────────

    def _init_fallback(self) -> None:
        from sklearn.ensemble import IsolationForest
        import random
        from shared.config import (
            VOLTAGE_MIN, VOLTAGE_MAX, CURRENT_MIN, CURRENT_MAX,
            TEMP_MIN, TEMP_MAX, HARMONIC_MIN, HARMONIC_MAX, LOAD_MIN, LOAD_MAX,
            ISOLATION_FOREST_CONTAMINATION, ISOLATION_FOREST_TRAINING_SAMPLES,
        )
        logger.info("[AnomalyDetector] Training fallback IsolationForest…")
        self.model = IsolationForest(contamination=ISOLATION_FOREST_CONTAMINATION, random_state=42)
        data = [[
            random.uniform(VOLTAGE_MIN, VOLTAGE_MAX),
            random.uniform(CURRENT_MIN, CURRENT_MAX),
            random.uniform(TEMP_MIN, TEMP_MAX),
            random.uniform(HARMONIC_MIN, HARMONIC_MAX),
            random.uniform(LOAD_MIN, LOAD_MAX),
        ] for _ in range(ISOLATION_FOREST_TRAINING_SAMPLES)]
        self.model.fit(data)
        logger.info("[AnomalyDetector] Fallback ready.")

    def _fallback_detect(self, telemetry: dict) -> bool:
        from shared.config import VOLTAGE_MIN, VOLTAGE_MAX, CURRENT_MIN, CURRENT_MAX
        from shared.config import TEMP_MIN, TEMP_MAX, HARMONIC_MIN, HARMONIC_MAX, LOAD_MIN, LOAD_MAX
        defaults = {
            "voltage":         (VOLTAGE_MIN + VOLTAGE_MAX) / 2,
            "current":         (CURRENT_MIN + CURRENT_MAX) / 2,
            "temperature":     (TEMP_MIN + TEMP_MAX) / 2,
            "harmonic_5th":    (HARMONIC_MIN + HARMONIC_MAX) / 2,
            "load_percentage": (LOAD_MIN + LOAD_MAX) / 2,
        }
        features = []
        for field in FEATURE_ORDER:
            val = telemetry.get(field)
            if val is None:
                features.append(defaults[field])
            elif field == "voltage" and float(val) < 10.0:
                features.append(defaults["voltage"])
            else:
                features.append(float(val))
        pred = self.model.predict(np.array([features]))[0]
        return bool(pred == -1)
