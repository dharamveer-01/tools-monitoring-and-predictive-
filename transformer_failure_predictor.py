"""
TransformerFailurePredictor — predicts probability of transformer failure
within the next 24 hours based on combined sensor trends.

Model: Logistic Regression + feature engineering (sklearn)
Input: aggregated statistics from last 20 readings
Output: failure probability (0.0 – 1.0) + risk classification

Key failure indicators:
  - Sustained high temperature (thermal stress)
  - Repeated voltage sags (insulation stress)
  - High harmonic distortion (winding stress)
  - Overload cycles (mechanical stress)
  - Rapid temperature changes (thermal cycling)

Usage:
    predictor = TransformerFailurePredictor()
    predictor.train()
    result = predictor.predict(history)
    # {'failure_probability': 0.73, 'risk': 'HIGH', 'top_indicators': [...]}
"""
import os
import sys
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import MODEL_SAVE_PATH
from shared.utils import get_logger, ensure_dir

logger = get_logger(__name__)

MODEL_FILE  = os.path.join(MODEL_SAVE_PATH, "transformer_failure_predictor.joblib")
SCALER_FILE = os.path.join(MODEL_SAVE_PATH, "transformer_failure_scaler.joblib")
WINDOW_SIZE = 20

FEATURE_NAMES = [
    "temp_mean", "temp_max", "temp_std", "temp_rate",
    "voltage_sag_count", "voltage_std", "voltage_min",
    "harmonic_mean", "harmonic_max", "harmonic_over8_count",
    "load_mean", "load_max", "load_over80_count",
    "current_mean", "current_max",
    "thermal_cycles",   # number of temp direction changes
    "combined_stress",  # weighted stress index
]


def _extract_features(history: list[dict]) -> list:
    """Extract transformer stress features from a window of readings."""
    temps    = [float(p.get("temperature",     62)) for p in history]
    voltages = [float(p.get("voltage",        230)) for p in history]
    harmonics= [float(p.get("harmonic_5th",     3)) for p in history]
    loads    = [float(p.get("load_percentage", 40)) for p in history]
    currents = [float(p.get("current",         15)) for p in history]

    t = np.array(temps)
    v = np.array(voltages)
    h = np.array(harmonics)
    lo= np.array(loads)
    c = np.array(currents)

    # Temperature features
    temp_rate = float(t[-1] - t[0]) / len(t)   # avg rate of change
    thermal_cycles = sum(
        1 for i in range(1, len(t) - 1)
        if (t[i] - t[i-1]) * (t[i+1] - t[i]) < 0
    )

    # Voltage features
    voltage_sag_count = int(np.sum(v < 210))

    # Harmonic features
    harm_over8 = int(np.sum(h > 8))

    # Load features
    load_over80 = int(np.sum(lo > 80))

    # Combined stress index (weighted)
    stress = (
        max(0, float(t.mean()) - 75) * 0.4 +
        voltage_sag_count * 2.0 +
        float(h.mean()) * 1.5 +
        max(0, float(lo.mean()) - 60) * 0.3
    )

    return [
        float(t.mean()), float(t.max()), float(t.std()), temp_rate,
        voltage_sag_count, float(v.std()), float(v.min()),
        float(h.mean()), float(h.max()), harm_over8,
        float(lo.mean()), float(lo.max()), load_over80,
        float(c.mean()), float(c.max()),
        thermal_cycles,
        stress,
    ]


class TransformerFailurePredictor:
    """
    Predicts transformer failure risk from recent telemetry history.
    """

    def __init__(self, failure_threshold: float = 0.5):
        self.threshold = failure_threshold
        self.model  = LogisticRegression(C=1.0, random_state=42, max_iter=500)
        self.scaler = StandardScaler()
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, n_sequences: int = 2000) -> dict:
        import random
        logger.info(f"[TransformerFailure] Generating {n_sequences} training sequences…")

        X, y = [], []
        for _ in range(n_sequences):
            will_fail = random.random() < 0.25

            history = []
            for i in range(WINDOW_SIZE):
                if will_fail:
                    # Stressed readings: high temp, voltage sags, harmonics
                    history.append({
                        "temperature":     random.uniform(80, 120),
                        "voltage":         random.uniform(195, 215),
                        "harmonic_5th":    random.uniform(7, 16),
                        "load_percentage": random.uniform(65, 92),
                        "current":         random.uniform(18, 28),
                    })
                else:
                    history.append({
                        "temperature":     random.uniform(50, 75),
                        "voltage":         random.uniform(220, 240),
                        "harmonic_5th":    random.uniform(1, 5),
                        "load_percentage": random.uniform(20, 60),
                        "current":         random.uniform(10, 18),
                    })

            X.append(_extract_features(history))
            y.append(1 if will_fail else 0)

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model.fit(X_train_s, y_train)
        self.is_trained = True

        probs = self.model.predict_proba(X_test_s)[:, 1]
        auc   = roc_auc_score(y_test, probs)
        logger.info(f"[TransformerFailure] AUC-ROC={auc:.3f}")
        return {"auc_roc": round(auc, 3), "n_train": len(X_train)}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, history: list[dict]) -> dict:
        if not self.is_trained:
            self.train()

        if len(history) < WINDOW_SIZE:
            return {
                "failure_probability": 0.0,
                "risk": "UNKNOWN",
                "insufficient_history": True,
                "readings_needed": WINDOW_SIZE - len(history),
            }

        window   = history[-WINDOW_SIZE:]
        features = np.array([_extract_features(window)])
        features_s = self.scaler.transform(features)
        prob = float(self.model.predict_proba(features_s)[0][1])

        risk = "CRITICAL" if prob > 0.7 else "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.3 else "LOW"

        # Top stress indicators
        raw = _extract_features(window)
        indicators = sorted(
            zip(FEATURE_NAMES, raw),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]

        return {
            "failure_probability": round(prob, 3),
            "risk":                risk,
            "will_fail":           prob >= self.threshold,
            "top_indicators":      [{"feature": n, "value": round(v, 2)} for n, v in indicators],
            "horizon_hours":       24,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        ensure_dir(MODEL_SAVE_PATH)
        joblib.dump(self.model,  MODEL_FILE)
        joblib.dump(self.scaler, SCALER_FILE)
        logger.info(f"[TransformerFailure] Saved to {MODEL_SAVE_PATH}")

    def load(self) -> bool:
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            self.model  = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            self.is_trained = True
            logger.info("[TransformerFailure] Loaded from disk.")
            return True
        return False
