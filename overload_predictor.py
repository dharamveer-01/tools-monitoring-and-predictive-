"""
OverloadPredictor — predicts whether a substation will overload
in the next N readings based on current trend.

Model: Random Forest Classifier (sklearn)
Input: rolling statistics of load, current, temperature over last 10 readings
Output: probability of overload in next 5 readings (0.0 – 1.0)

An "overload" is defined as load_percentage > 80% OR current > 22A.

Usage:
    predictor = OverloadPredictor()
    predictor.train()
    result = predictor.predict(history)
    # {'overload_probability': 0.82, 'will_overload': True, 'horizon': 5}
"""
import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import MODEL_SAVE_PATH, LOAD_MAX, CURRENT_MAX
from shared.utils import get_logger, ensure_dir

logger = get_logger(__name__)

MODEL_FILE   = os.path.join(MODEL_SAVE_PATH, "overload_predictor.joblib")
WINDOW_SIZE  = 10    # readings to look back
HORIZON      = 5     # readings to look ahead for overload
OVERLOAD_THRESHOLD_LOAD    = 80.0
OVERLOAD_THRESHOLD_CURRENT = 22.0


def _extract_window_features(history: list[dict]) -> list:
    """
    Extract trend features from a window of telemetry readings.
    history: list of dicts, oldest first, length = WINDOW_SIZE
    """
    loads    = [float(p.get("load_percentage", 40)) for p in history]
    currents = [float(p.get("current", 15))         for p in history]
    temps    = [float(p.get("temperature", 62))      for p in history]

    def stats(vals):
        arr = np.array(vals)
        return [
            float(arr.mean()),
            float(arr.std()),
            float(arr.max()),
            float(arr[-1]),                          # latest value
            float(arr[-1] - arr[0]),                 # total change
            float(arr[-1] - arr[-2]) if len(arr) > 1 else 0.0,  # last delta
        ]

    return stats(loads) + stats(currents) + stats(temps)   # 18 features


class OverloadPredictor:
    """
    Predicts overload risk from recent telemetry history.
    """

    def __init__(self, overload_probability_threshold: float = 0.6):
        self.threshold = overload_probability_threshold
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"
        )
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, n_sequences: int = 1500) -> dict:
        """Generate synthetic sequences and train."""
        import random, math
        logger.info(f"[OverloadPredictor] Generating {n_sequences} training sequences…")

        X, y = [], []
        for _ in range(n_sequences):
            # Decide if this sequence leads to overload
            will_overload = random.random() < 0.35

            history = []
            base_load = random.uniform(30, 60)
            if will_overload:
                # Gradually increasing load
                for i in range(WINDOW_SIZE):
                    load = base_load + i * random.uniform(2, 5) + random.gauss(0, 2)
                    history.append({
                        "load_percentage": min(load, 95),
                        "current":         10 + load / 10 + random.gauss(0, 0.5),
                        "temperature":     55 + load * 0.3 + random.gauss(0, 2),
                    })
            else:
                # Stable load
                for i in range(WINDOW_SIZE):
                    load = base_load + random.gauss(0, 3)
                    history.append({
                        "load_percentage": max(10, min(load, 65)),
                        "current":         10 + load / 15 + random.gauss(0, 0.5),
                        "temperature":     55 + load * 0.2 + random.gauss(0, 2),
                    })

            features = _extract_window_features(history)
            X.append(features)
            y.append(1 if will_overload else 0)

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"[OverloadPredictor] F1={report['1']['f1-score']:.3f}")
        return {"f1": round(report["1"]["f1-score"], 3), "n_train": len(X_train)}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, history: list[dict]) -> dict:
        """
        Predict overload risk from recent history.
        history: list of telemetry dicts, oldest first, min length = WINDOW_SIZE
        """
        if not self.is_trained:
            self.train()

        if len(history) < WINDOW_SIZE:
            return {
                "overload_probability": 0.0,
                "will_overload":        False,
                "horizon":              HORIZON,
                "insufficient_history": True,
                "readings_needed":      WINDOW_SIZE - len(history),
            }

        window   = history[-WINDOW_SIZE:]
        features = np.array([_extract_window_features(window)])
        prob     = float(self.model.predict_proba(features)[0][1])

        return {
            "overload_probability": round(prob, 3),
            "will_overload":        prob >= self.threshold,
            "horizon":              HORIZON,
            "risk_level":           "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        ensure_dir(MODEL_SAVE_PATH)
        joblib.dump(self.model, MODEL_FILE)
        logger.info(f"[OverloadPredictor] Saved to {MODEL_FILE}")

    def load(self) -> bool:
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
            self.is_trained = True
            logger.info("[OverloadPredictor] Loaded from disk.")
            return True
        return False
