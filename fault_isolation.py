"""
FaultIsolation — identifies and classifies the type of fault in a telemetry packet.

Returns a structured fault report that can be used by the alert system
and the root-cause explainability engine.
"""
from typing import Dict, Any

from shared.config import (
    VOLTAGE_MIN, VOLTAGE_MAX,
    TEMP_MAX,
    HARMONIC_MAX,
    LOAD_MAX,
    CURRENT_MAX,
)


FAULT_RULES = [
    {
        "name": "Overheat",
        "check": lambda p: p["temperature"] > TEMP_MAX + 10,
        "severity": "HIGH",
        "description": "Transformer/equipment temperature exceeds safe operating limit.",
    },
    {
        "name": "Voltage Sag",
        "check": lambda p: p["voltage"] < VOLTAGE_MIN - 15,
        "severity": "HIGH",
        "description": "Voltage dropped significantly below nominal range.",
    },
    {
        "name": "Voltage Surge",
        "check": lambda p: p["voltage"] > VOLTAGE_MAX + 15,
        "severity": "MEDIUM",
        "description": "Voltage exceeded safe upper limit.",
    },
    {
        "name": "Overload",
        "check": lambda p: p["load_percentage"] > LOAD_MAX + 20,
        "severity": "HIGH",
        "description": "Load percentage critically high — risk of equipment damage.",
    },
    {
        "name": "Harmonic Distortion",
        "check": lambda p: p["harmonic_5th"] > HARMONIC_MAX + 3,
        "severity": "MEDIUM",
        "description": "5th harmonic distortion exceeds IEEE 519 limits.",
    },
    {
        "name": "Overcurrent",
        "check": lambda p: p["current"] > CURRENT_MAX + 5,
        "severity": "HIGH",
        "description": "Current draw exceeds rated capacity.",
    },
]


def isolate_faults(packet: dict) -> Dict[str, Any]:
    """
    Analyse a telemetry packet and return a fault isolation report.

    Returns:
        {
            "faults_detected": [{"name": ..., "severity": ..., "description": ...}],
            "fault_count": int,
            "highest_severity": "HIGH" | "MEDIUM" | "LOW" | "NONE"
        }
    """
    detected = []
    for rule in FAULT_RULES:
        try:
            if rule["check"](packet):
                detected.append({
                    "name":        rule["name"],
                    "severity":    rule["severity"],
                    "description": rule["description"],
                })
        except (KeyError, TypeError):
            continue

    severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    highest = "NONE"
    for f in detected:
        if severity_order.get(f["severity"], 0) > severity_order[highest]:
            highest = f["severity"]

    return {
        "faults_detected":  detected,
        "fault_count":      len(detected),
        "highest_severity": highest,
    }
