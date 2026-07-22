"""
SelfHealingEngine — gradually restores load to recovering substations.

Logic:
  1. Tracks which substations were previously Critical.
  2. When a substation's health score rises above RECOVERY_THRESHOLD,
     it enters "recovering" state.
  3. Each healing cycle, load is restored by RECOVERY_STEP % until
     the substation reaches its fair share.
  4. If health drops again, recovery is paused.
"""
import threading
import time
from typing import Dict

from shared.config import (
    RECOVERY_THRESHOLD,
    RECOVERY_STEP,
    HEALING_INTERVAL,
    CRITICAL_LOAD_FLOOR,
)
from shared.utils import get_logger

logger = get_logger(__name__)


class SelfHealingEngine:
    def __init__(self, load_balancer):
        """
        load_balancer: reference to the LoadBalancer instance so we can
                       directly adjust its load_distribution dict.
        """
        self.balancer = load_balancer
        self._recovering: Dict[str, bool] = {}   # sub_id → is_recovering
        self._lock = threading.Lock()
        self._running = False

    # ── public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background healing loop."""
        self._running = True
        t = threading.Thread(target=self._healing_loop, daemon=True)
        t.start()
        logger.info("Self-healing engine started.")

    def stop(self) -> None:
        self._running = False

    def notify_health(self, sub_id: str, health_score: float, risk_level: str) -> None:
        """
        Called by the socket server on every telemetry packet.
        Marks substations as recovering when they improve.
        """
        with self._lock:
            if risk_level == "Critical":
                self._recovering[sub_id] = False
            elif health_score >= RECOVERY_THRESHOLD:
                if not self._recovering.get(sub_id, True):
                    logger.info(f"[HEALING] {sub_id} health improved to {health_score:.1f} — starting recovery.")
                self._recovering[sub_id] = True

    # ── private ───────────────────────────────────────────────────────────────

    def _healing_loop(self) -> None:
        while self._running:
            time.sleep(HEALING_INTERVAL)
            self._run_healing_cycle()

    def _run_healing_cycle(self) -> None:
        with self._lock:
            recovering_subs = [s for s, r in self._recovering.items() if r]

        if not recovering_subs:
            return

        active = list(self.balancer.load_distribution.keys())
        if not active:
            return

        fair_share = 100.0 / len(active)

        for sub_id in recovering_subs:
            current_load = self.balancer.load_distribution.get(sub_id, CRITICAL_LOAD_FLOOR)
            if current_load < fair_share - 0.5:
                new_load = min(current_load + RECOVERY_STEP, fair_share)
                self.balancer.load_distribution[sub_id] = round(new_load, 1)
                logger.info(
                    f"[HEALING] {sub_id} load restored: {current_load:.1f}% → {new_load:.1f}%"
                )
            else:
                # Fully recovered
                with self._lock:
                    self._recovering[sub_id] = False
                logger.info(f"[HEALING] {sub_id} fully recovered to {fair_share:.1f}%.")
