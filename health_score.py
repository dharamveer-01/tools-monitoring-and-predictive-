"""
Health score calculator.

Handles real USB sensor data where some fields may be None
(e.g. a sensor that only measures temperature and voltage).
Only applies penalties for fields that are actually present.
"""


def calculate_health_score(telemetry: dict, is_anomaly: bool) -> tuple[float, str]:
    """
    Calculate a health score (0–100) from telemetry and anomaly flag.

    Fields that are None (sensor didn't send them) are skipped — no penalty,
    no false positives from missing data.

    Returns (score: float, status: str)
    """
    score = 100.0

    temp     = telemetry.get("temperature")
    voltage  = telemetry.get("voltage")
    harmonic = telemetry.get("harmonic_5th")
    load     = telemetry.get("load_percentage")

    # If voltage is in phone battery range (< 10V), skip voltage penalty —
    # it's raw battery voltage from the old ADB client, not grid voltage
    voltage_is_grid = voltage is not None and voltage >= 10.0

    # Temperature penalty
    if temp is not None and temp > 85:
        score -= (temp - 85) * 1.5

    # Voltage penalty — only apply when voltage is in grid range (≥10V)
    if voltage_is_grid and (voltage < 200 or voltage > 250):
        score -= 20

    # Harmonic distortion penalty
    if harmonic is not None and harmonic > 8:
        score -= min((harmonic - 8) * 3, 30)

    # Load penalty (overload)
    if load is not None and load > 80:
        score -= (load - 80) * 0.5

    # ML anomaly penalty
    if is_anomaly:
        score -= 30

    score = max(0.0, min(100.0, score))

    if score >= 80:
        status = "Healthy"
    elif score >= 50:
        status = "Warning"
    else:
        status = "Critical"

    return round(score, 1), status
