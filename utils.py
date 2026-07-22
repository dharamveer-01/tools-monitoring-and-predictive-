"""
Shared utility helpers used across the project.
"""
import json
import logging
import os
from datetime import datetime


# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Telemetry helpers ────────────────────────────────────────────────────────

def validate_telemetry(packet: dict) -> bool:
    """
    Return True if the packet has the minimum required fields.

    Rules:
    - substation_id and timestamp must be present and non-empty strings.
    - Numeric fields (voltage, current, temperature, harmonic_5th, load_percentage)
      must be present as keys, but their values may be None (sensor didn't send them)
      or a numeric value.  A non-numeric, non-None value is rejected.
    """
    if not isinstance(packet, dict):
        return False

    # Identity fields must be present and non-empty
    for field in ("substation_id", "timestamp"):
        if not packet.get(field):
            return False

    # Numeric fields must exist as keys; None is allowed (partial sensor data)
    numeric_fields = ["voltage", "current", "temperature", "harmonic_5th", "load_percentage"]
    for field in numeric_fields:
        if field not in packet:
            return False
        val = packet[field]
        if val is not None:
            try:
                float(val)
            except (TypeError, ValueError):
                return False

    return True


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.utcnow().isoformat()


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def safe_json_loads(raw: str) -> dict | None:
    """Parse JSON without raising; returns None on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
