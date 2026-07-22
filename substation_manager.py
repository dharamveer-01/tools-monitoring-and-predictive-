"""
SubstationManager — single source of truth for all substation state.

Aggregates:
  - live telemetry (from TelemetryManager)
  - health scores
  - load distribution (from LoadBalancer)
  - fault isolation reports
  - alert history

Used by the API layer to serve the /state endpoint.
"""
import threading
from typing import Dict, Any

from shared.utils import get_logger

logger = get_logger(__name__)


class SubstationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._health: Dict[str, dict] = {}
        self._fault_reports: Dict[str, dict] = {}

    # ── write ─────────────────────────────────────────────────────────────────

    def update_health(self, sub_id: str, health_record: dict) -> None:
        with self._lock:
            self._health[sub_id] = health_record

    def update_fault_report(self, sub_id: str, report: dict) -> None:
        with self._lock:
            self._fault_reports[sub_id] = report

    def remove(self, sub_id: str) -> None:
        with self._lock:
            self._health.pop(sub_id, None)
            self._fault_reports.pop(sub_id, None)

    # ── read ──────────────────────────────────────────────────────────────────

    def get_health(self, sub_id: str) -> dict:
        with self._lock:
            return dict(self._health.get(sub_id, {}))

    def get_all_health(self) -> Dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._health.items()}

    def get_fault_report(self, sub_id: str) -> dict:
        with self._lock:
            return dict(self._fault_reports.get(sub_id, {}))

    def get_all_fault_reports(self) -> Dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._fault_reports.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Return a compact status summary for all substations."""
        with self._lock:
            summary = {}
            for sub_id, h in self._health.items():
                summary[sub_id] = {
                    "health_score":     h.get("health_score", 0),
                    "risk_level":       h.get("risk_level", "Unknown"),
                    "anomaly_detected": h.get("anomaly_detected", False),
                }
            return summary
