"""
TelemetryGenerator — produces realistic electrical telemetry packets.

Normal operating ranges (from config):
  Voltage   : 220–240 V
  Current   : 10–20 A
  Temp      : 50–75 °C
  Harmonic  : 1–5 %
  Load      : 20–60 %

Gaussian noise is added to make the stream look like real sensor data.
"""
import random
import math
from datetime import datetime

from shared.config import (
    VOLTAGE_MIN, VOLTAGE_MAX,
    CURRENT_MIN, CURRENT_MAX,
    TEMP_MIN, TEMP_MAX,
    HARMONIC_MIN, HARMONIC_MAX,
    LOAD_MIN, LOAD_MAX,
)


class TelemetryGenerator:
    """Generates one telemetry packet per call with optional drift simulation."""

    def __init__(self, substation_id: str):
        self.substation_id = substation_id
        self._tick = 0                  # used for slow sinusoidal drift
        self._base_temp = (TEMP_MIN + TEMP_MAX) / 2
        self._base_load = (LOAD_MIN + LOAD_MAX) / 2

    def generate(self) -> dict:
        """Return a normal (healthy) telemetry packet."""
        self._tick += 1

        # Slow sinusoidal drift to simulate load cycles
        drift = math.sin(self._tick * 0.05) * 5

        voltage     = random.gauss((VOLTAGE_MIN + VOLTAGE_MAX) / 2, 3)
        current     = random.gauss((CURRENT_MIN + CURRENT_MAX) / 2, 1.5)
        temperature = random.gauss(self._base_temp + drift * 0.3, 2)
        harmonic    = random.gauss((HARMONIC_MIN + HARMONIC_MAX) / 2, 0.5)
        load        = random.gauss(self._base_load + drift, 3)

        # Clamp to realistic bounds
        voltage     = max(VOLTAGE_MIN - 5, min(VOLTAGE_MAX + 5, voltage))
        current     = max(CURRENT_MIN - 2, min(CURRENT_MAX + 2, current))
        temperature = max(TEMP_MIN - 5,    min(TEMP_MAX + 5,    temperature))
        harmonic    = max(0.5,             min(HARMONIC_MAX + 1, harmonic))
        load        = max(LOAD_MIN - 5,    min(LOAD_MAX + 5,     load))

        return {
            "substation_id":  self.substation_id,
            "timestamp":      datetime.now().isoformat(),
            "voltage":        round(voltage, 2),
            "current":        round(current, 2),
            "temperature":    round(temperature, 2),
            "harmonic_5th":   round(harmonic, 2),
            "load_percentage": round(load, 2),
        }
