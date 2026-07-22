"""
HealthScoreModel — ML-based health score predictor.

Unlike the rule-based health_score.py, this model LEARNS the relationship
between telemetry features and health outcomes from historical data.

Model: Gradient Boosting Regressor (sklearn)
Input: 5 telemetry features + 6 derived features
Output: predicted health score (0–100)

Advantage over rule-based:
  - Captures non-linear interactions between features
  - Can be retrained as new fault patterns are observed
  - Provides feature importance rankings
"""
import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import (
    MODEL_SAVE_PATH,
    VOLTAGE_MIN, VOLTAGE_MAX, CURRENT_MIN, CURRENT_MAX,
    TEMP_MIN, TEMP_MAX, HARMONIC_MIN, HARMONIC_MAX, LOAD_MIN, LOAD_MAX,
)
from shared.utils import get_logger, ensure_dir
from ml.health_score import calculate_health_score

logger = get_logger(__name__)

MODEL_FILE = os.path.join(MODEL_SAVE_PATH, "health_score_model.joblib")

FEATURE_NAMES = [
    "voltage", "current", "temperature", "harmonic_5th", "load_percentage",
    "temp_over_limit", "voltage_deviation", "harmonic_over_limit",
    "load_over_limit", "combined_stress", "is_anomaly_flag",
]


def _extract_features(packet: dict, is_anomaly: bool = False) -> list:
    v  = float(packet.get("voltage",         230))
    c  = float(packet.get("current",          15))
    t  = float(packet.get("temperature",      62))
    h  = float(packet.get("harmonic_5th",      3))
    lo = float(packet.get("load_percentage",  40))

    temp_over      = max(0, t - 85)
    volt_dev       = abs(v - 230) / 10.0
    harm_over      = max(0, h - 8)
    load_over      = max(0, lo - 80)
    combined       = temp_over * 0.4 + volt_dev * 0.2 + harm_over * 0.2 + load_over * 0.2
    anomaly_flag   = 1.0 if is_anomaly else 0.0

    return [v, c, t, h, lo, temp_over, volt_dev, harm_over, load_over, combined, anomaly_flag]


class HealthScoreModel:
    """
    ML-based health score predictor trained on rule-based labels.

    Usage:
        model = HealthScoreModel()
        model.train()
        score = model.predict(packet, is_anomaly=False)
        model.save()
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, n_samples: int = 2000) -> dict:
        """
        Generate synthetic training data using the rule-based scorer as ground truth,
        then train the ML model to replicate and generalise it.
        """
        import random
        logger.info(f"[HealthScoreModel] Generating {n_samples} training samples…")

        X, y = [], []
        for _ in range(n_samples):
            # Mix of normal and faulty readings
            if random.random() < 0.3:
                packet = {
                    "voltage":         random.uniform(160, 200),
                    "current":         random.uniform(20, 35),
                    "temperature":     random.uniform(80, 130),
                    "harmonic_5th":    random.uniform(8, 18),
                    "load_percentage": random.uniform(70, 98),
                }
                is_anomaly = True
            else:
                packet = {
                    "voltage":         random.uniform(VOLTAGE_MIN, VOLTAGE_MAX),
                    "current":         random.uniform(CURRENT_MIN, CURRENT_MAX),
                    "temperature":     random.uniform(TEMP_MIN, TEMP_MAX),
                    "harmonic_5th":    random.uniform(HARMONIC_MIN, HARMONIC_MAX),
                    "load_percentage": random.uniform(LOAD_MIN, LOAD_MAX),
                }
                is_anomaly = random.random() < 0.05

            score, _ = calculate_health_score(packet, is_anomaly)
            X.append(_extract_features(packet, is_anomaly))
            y.append(score)

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        logger.info(f"[HealthScoreModel] MAE={mae:.2f}  R²={r2:.4f}")

        return {"mae": round(mae, 2), "r2": round(r2, 4), "n_train": len(X_train)}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, packet: dict, is_anomaly: bool = False) -> float:
        if not self.is_trained:
            self.train()
        features = np.array([_extract_features(packet, is_anomaly)])
        score = float(self.model.predict(features)[0])
        return round(max(0.0, min(100.0, score)), 1)

    def feature_importance(self) -> dict:
        if not self.is_trained:
            return {}
        return {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, self.model.feature_importances_)
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        ensure_dir(MODEL_SAVE_PATH)
        joblib.dump(self.model, MODEL_FILE)
        logger.info(f"[HealthScoreModel] Saved to {MODEL_FILE}")

    def load(self) -> bool:
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
            self.is_trained = True
            logger.info("[HealthScoreModel] Loaded from disk.")
            return True
        return False
