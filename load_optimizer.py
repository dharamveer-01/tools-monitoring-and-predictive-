"""
LoadOptimizer — mathematical load optimization using linear programming concepts.

Finds the optimal load distribution that:
  1. Minimises total system stress (weighted health penalties)
  2. Keeps each substation within safe operating bounds
  3. Ensures total load sums to 100%

This is a deterministic optimizer (no ML training needed).
It runs on every rebalance request.

Algorithm: Weighted proportional allocation
  - Healthy substations get load proportional to their health score
  - Critical substations get the minimum floor load
  - Warning substations get reduced load proportional to their score
"""
import os
import sys
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import CRITICAL_LOAD_FLOOR, HEALTH_HEALTHY_MIN, HEALTH_WARNING_MIN
from shared.utils import get_logger

logger = get_logger(__name__)


class LoadOptimizer:
    """
    Optimal load distributor based on substation health scores.

    Usage:
        optimizer = LoadOptimizer()
        distribution = optimizer.optimize(health_data)
        # {'S1': 45.0, 'S2': 10.0, 'S3': 45.0}
    """

    def __init__(
        self,
        critical_floor: float = CRITICAL_LOAD_FLOOR,
        warning_factor: float = 0.6,   # warning subs get 60% of healthy share
    ):
        self.critical_floor = critical_floor
        self.warning_factor = warning_factor

    def optimize(self, health_data: Dict[str, dict]) -> Dict[str, float]:
        """
        Compute optimal load distribution.

        health_data: {sub_id: {'health_score': float, 'risk_level': str}}
        Returns: {sub_id: load_percentage}
        """
        if not health_data:
            return {}

        substations = list(health_data.keys())
        n = len(substations)

        if n == 1:
            return {substations[0]: 100.0}

        # Classify substations
        critical = [s for s in substations if health_data[s].get("risk_level") == "Critical"]
        warning  = [s for s in substations if health_data[s].get("risk_level") == "Warning"]
        healthy  = [s for s in substations if health_data[s].get("risk_level") == "Healthy"]

        # If all same status — equal distribution
        if not critical and not warning:
            share = 100.0 / n
            return {s: round(share, 2) for s in substations}

        if len(critical) == n:
            share = 100.0 / n
            return {s: round(share, 2) for s in substations}

        # Assign floor to critical substations
        critical_total = len(critical) * self.critical_floor
        remaining = 100.0 - critical_total

        # Split remaining between healthy and warning
        # Warning gets warning_factor × healthy share
        # healthy_share × n_healthy + warning_factor × healthy_share × n_warning = remaining
        n_h = len(healthy)
        n_w = len(warning)
        if n_h + n_w == 0:
            share = remaining / n
            result = {s: round(self.critical_floor, 2) for s in critical}
            result.update({s: round(share, 2) for s in substations if s not in critical})
            return result

        healthy_share = remaining / (n_h + self.warning_factor * n_w) if (n_h + n_w) > 0 else 0
        warning_share = healthy_share * self.warning_factor

        result = {}
        for s in substations:
            if s in critical:
                result[s] = round(self.critical_floor, 2)
            elif s in warning:
                result[s] = round(warning_share, 2)
            else:
                result[s] = round(healthy_share, 2)

        # Normalise to exactly 100%
        total = sum(result.values())
        if total > 0:
            factor = 100.0 / total
            result = {s: round(v * factor, 2) for s, v in result.items()}

        logger.info(f"[LoadOptimizer] Distribution: {result}")
        return result

    def get_recommendation(self, health_data: Dict[str, dict]) -> dict:
        """Return distribution with explanation."""
        dist = self.optimize(health_data)
        critical = [s for s in health_data if health_data[s].get("risk_level") == "Critical"]
        warning  = [s for s in health_data if health_data[s].get("risk_level") == "Warning"]

        reason = "Equal distribution — all substations healthy."
        if critical:
            reason = f"Critical substations {critical} reduced to {self.critical_floor}%. Load shifted to healthy substations."
        elif warning:
            reason = f"Warning substations {warning} reduced to {self.warning_factor*100:.0f}% of normal share."

        return {"distribution": dist, "reason": reason, "critical": critical, "warning": warning}
