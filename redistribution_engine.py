"""
RedistributionEngine — manages the full load redistribution lifecycle.

Responsibilities:
  1. Detect when redistribution is needed (state change)
  2. Call LoadOptimizer to compute new distribution
  3. Apply the distribution smoothly (gradual transition, not instant jump)
  4. Log redistribution events for audit trail
  5. Prevent oscillation (cooldown between redistributions)

This sits above LoadOptimizer and LoadBalancer — it's the decision layer.
"""
import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml_models.load_balancing.load_optimizer import LoadOptimizer
from shared.config import CRITICAL_LOAD_FLOOR
from shared.utils import get_logger

logger = get_logger(__name__)

REDISTRIBUTION_COOLDOWN = 15   # seconds between redistributions per substation
TRANSITION_STEPS        = 5    # number of steps to gradually move load
TRANSITION_INTERVAL     = 2.0  # seconds between transition steps


class RedistributionEngine:
    """
    Full load redistribution lifecycle manager.

    Usage:
        engine = RedistributionEngine(load_balancer)
        engine.check_and_redistribute(health_data)
    """

    def __init__(self, load_balancer):
        self.balancer   = load_balancer
        self.optimizer  = LoadOptimizer()
        self._lock      = threading.Lock()
        self._last_redistribution: Dict[str, float] = {}   # sub_id → timestamp
        self._history:  List[dict] = []
        self._previous_statuses: Dict[str, str] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def check_and_redistribute(self, health_data: Dict[str, dict]) -> dict | None:
        """
        Check if redistribution is needed and apply it.
        Returns the redistribution event dict if triggered, else None.
        """
        if not health_data:
            return None

        # Detect status changes
        changed = self._detect_changes(health_data)
        if not changed:
            return None

        # Check cooldown
        now = time.time()
        for sub_id in changed:
            last = self._last_redistribution.get(sub_id, 0)
            if now - last < REDISTRIBUTION_COOLDOWN:
                logger.debug(f"[Redistribution] Cooldown active for {sub_id}, skipping.")
                return None

        # Compute new distribution
        recommendation = self.optimizer.get_recommendation(health_data)
        new_dist = recommendation["distribution"]

        # Apply gradually in background
        threading.Thread(
            target=self._apply_gradually,
            args=(new_dist,),
            daemon=True,
        ).start()

        # Record event
        event = {
            "timestamp":      datetime.now().isoformat(),
            "trigger":        changed,
            "new_distribution": new_dist,
            "reason":         recommendation["reason"],
            "critical":       recommendation["critical"],
            "warning":        recommendation["warning"],
        }
        with self._lock:
            self._history.append(event)
            if len(self._history) > 100:
                self._history.pop(0)
            for sub_id in changed:
                self._last_redistribution[sub_id] = now

        logger.info(
            f"[Redistribution] Triggered by {changed}. "
            f"New dist: {new_dist}. Reason: {recommendation['reason']}"
        )
        return event

    def get_history(self, limit: int = 20) -> List[dict]:
        with self._lock:
            return list(reversed(self._history[-limit:]))

    # ── Private ───────────────────────────────────────────────────────────────

    def _detect_changes(self, health_data: Dict[str, dict]) -> List[str]:
        """Return list of substations whose status changed since last check."""
        changed = []
        for sub_id, h in health_data.items():
            current_status = h.get("risk_level", "Unknown")
            previous_status = self._previous_statuses.get(sub_id)
            if previous_status != current_status:
                changed.append(sub_id)
                self._previous_statuses[sub_id] = current_status
        return changed

    def _apply_gradually(self, target: Dict[str, float]) -> None:
        """Smoothly transition load distribution to target over several steps."""
        current = dict(self.balancer.load_distribution)
        for step in range(1, TRANSITION_STEPS + 1):
            alpha = step / TRANSITION_STEPS   # 0.2, 0.4, 0.6, 0.8, 1.0
            for sub_id, target_load in target.items():
                current_load = current.get(sub_id, target_load)
                interpolated = current_load + alpha * (target_load - current_load)
                self.balancer.load_distribution[sub_id] = round(interpolated, 2)
            time.sleep(TRANSITION_INTERVAL)
        logger.info(f"[Redistribution] Transition complete: {self.balancer.load_distribution}")
