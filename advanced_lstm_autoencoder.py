import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

class AdvancedAnomalyDetector:
    """
    Advanced AI Model: LSTM Autoencoder
    
    Why this is better than standard IsolationForest:
    1. Time-Series Awareness: LSTMs remember historical sequence patterns, 
       meaning it catches gradual fatigue (e.g., slow thermal rise + current drop) 
       instead of just instantaneous spikes.
    2. Reconstruction Error: It learns what "Normal" hardware telemetry looks like.
       If the hardware sends a sequence it can't reconstruct well, the error spikes,
       and it flags an anomaly.
    """
    def __init__(self, sequence_length=10, n_features=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        self.threshold = 0.5 # Dynamic thresholding applied later

    def _build_model(self):
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(self.sequence_length, self.n_features), return_sequences=False),
            RepeatVector(self.sequence_length),
            LSTM(32, activation='relu', return_sequences=True),
            TimeDistributed(Dense(self.n_features))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_on_baseline(self, healthy_history_data, epochs=10):
        """ Train on healthy hardware baselines """
        # Data should be shaped (samples, sequence_length, n_features)
        self.model.fit(healthy_history_data, healthy_history_data, epochs=epochs, batch_size=16, verbose=1)
        
        # Calculate max reconstruction error on healthy data to set a threshold
        preds = self.model.predict(healthy_history_data)
        errors = np.mean(np.square(preds - healthy_history_data), axis=2)
        self.threshold = np.max(errors) * 1.5 # 50% tolerance buffer

    def detect_streaming(self, recent_sequence):
        """ 
        Predict on real-time sliding window telemetry.
        Returns (is_anomaly: bool, anomaly_score_float)
        """
        # Shape to (1, sequence_length, n_features)
        seq = np.array(recent_sequence).reshape(1, self.sequence_length, self.n_features)
        pred = self.model.predict(seq, verbose=0)
        
        error = np.mean(np.square(pred[0] - seq[0]))
        is_anomaly = float(error) > self.threshold
        
        # Normalize score 0-1 for dashboard mapping
        score = min(float(error) / (self.threshold * 2), 1.0)
        
        return is_anomaly, score
