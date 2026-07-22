"""
FaultSimulator — injects realistic electrical fault patterns into telemetry.

Supported fault types:
  overheat        — temperature spike (transformer overheating)
  overload        — load + current surge
  voltage_sag     — voltage drops below safe range
  harmonic_dist   — harmonic distortion spike
  combined        — multiple faults simultaneously (worst case)

Usage:
    fs = FaultSimulator(fault_probability=0.3)
    packet = fs.maybe_inject(base_packet)
"""
import random
from shared.config import (
    FAULT_VOLTAGE_LOW, FAULT_VOLTAGE_HIGH,
    FAULT_TEMP_HIGH,
    FAULT_HARMONIC_HIGH,
    FAULT_LOAD_HIGH,
)


FAULT_TYPES = ["overheat", "overload", "voltage_sag", "harmonic_dist", "combined"]


class FaultSimulator:
    def __init__(self, fault_probability: float = 0.3):
        """
        fault_probability: chance (0–1) that any given packet is faulted.
        """
        self.fault_probability = fault_probability
        self._active_fault: str | None = None
        self._fault_duration = 0          # remaining ticks for current fault

    def maybe_inject(self, packet: dict) -> dict:
        """
        Possibly inject a fault into *packet* (mutates a copy).
        Returns the (possibly modified) packet and a fault_type string or None.
        """
        p = dict(packet)

        # Decide whether to start a new fault
        if self._fault_duration <= 0:
            if random.random() < self.fault_probability:
                self._active_fault = random.choice(FAULT_TYPES)
                self._fault_duration = random.randint(3, 8)   # lasts 3–8 ticks
            else:
                self._active_fault = None

        if self._active_fault:
            self._fault_duration -= 1
            p = self._apply_fault(p, self._active_fault)
            p["fault_type"] = self._active_fault
        else:
            p["fault_type"] = None

        return p

    # ── private ───────────────────────────────────────────────────────────────

    def _apply_fault(self, p: dict, fault_type: str) -> dict:
        if fault_type == "overheat":
            p["temperature"] = round(random.uniform(FAULT_TEMP_HIGH, FAULT_TEMP_HIGH + 25), 2)

        elif fault_type == "overload":
            p["load_percentage"] = round(random.uniform(FAULT_LOAD_HIGH, 98), 2)
            p["current"]         = round(random.uniform(22, 30), 2)

        elif fault_type == "voltage_sag":
            p["voltage"] = round(random.uniform(FAULT_VOLTAGE_LOW, FAULT_VOLTAGE_HIGH), 2)

        elif fault_type == "harmonic_dist":
            p["harmonic_5th"] = round(random.uniform(FAULT_HARMONIC_HIGH, 18), 2)

        elif fault_type == "combined":
            p["voltage"]         = round(random.uniform(FAULT_VOLTAGE_LOW, FAULT_VOLTAGE_HIGH), 2)
            p["temperature"]     = round(random.uniform(FAULT_TEMP_HIGH, FAULT_TEMP_HIGH + 20), 2)
            p["harmonic_5th"]    = round(random.uniform(FAULT_HARMONIC_HIGH, 16), 2)
            p["load_percentage"] = round(random.uniform(FAULT_LOAD_HIGH, 95), 2)

        return p
