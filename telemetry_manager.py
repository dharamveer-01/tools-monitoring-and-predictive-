"""
TelemetryManager — thread-safe in-memory store for live substation state.

Keeps:
  - latest telemetry packet per substation
  - rolling history window (configurable length) per substation
  - connection timestamps
"""
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

from shared.config import HISTORY_WINDOW


class TelemetryManager:
    def __init__(self, history_window: int = HISTORY_WINDOW):
        self._lock = threading.Lock()
        self._latest: Dict[str, dict] = {}
        self._history: Dict[str, deque] = {}
        self._connected_at: Dict[str, str] = {}
        self._history_window = history_window

    # ── write ─────────────────────────────────────────────────────────────────

    def update(self, packet: dict) -> None:
        """Store the latest packet and append to history."""
        sub_id = packet["substation_id"]
        with self._lock:
            self._latest[sub_id] = packet
            if sub_id not in self._history:
                self._history[sub_id] = deque(maxlen=self._history_window)
                self._connected_at[sub_id] = datetime.now().isoformat()
            self._history[sub_id].append(packet)

    def remove(self, sub_id: str) -> None:
        """Remove a substation (called on disconnect)."""
        with self._lock:
            self._latest.pop(sub_id, None)
            self._history.pop(sub_id, None)
            self._connected_at.pop(sub_id, None)

    # ── read ──────────────────────────────────────────────────────────────────

    def get_latest(self, sub_id: str) -> Optional[dict]:
        with self._lock:
            return dict(self._latest.get(sub_id, {}))

    def get_all_latest(self) -> Dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._latest.items()}

    def get_history(self, sub_id: str) -> List[dict]:
        with self._lock:
            return list(self._history.get(sub_id, []))

    def get_active_substations(self) -> List[str]:
        with self._lock:
            return list(self._latest.keys())

    def get_connection_info(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._connected_at)

    def substation_count(self) -> int:
        with self._lock:
            return len(self._latest)
