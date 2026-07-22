"""
ReinforcementAgent — Q-Learning agent for adaptive load balancing.

The agent learns from experience which load distributions lead to
better system health outcomes over time.

State:  health scores of all substations (discretised)
Action: load adjustment for each substation (-5%, 0%, +5%)
Reward: improvement in average system health score

This is a tabular Q-learning implementation (no neural network needed).
The Q-table is saved to disk and improves with every episode.

Usage:
    agent = ReinforcementAgent(n_substations=3)
    agent.load()                              # load existing Q-table if available
    action = agent.choose_action(state)       # get load adjustments
    agent.update(state, action, reward, next_state)  # learn from outcome
    agent.save()                              # persist Q-table
"""
import os
import sys
import numpy as np
import joblib
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import MODEL_SAVE_PATH, CRITICAL_LOAD_FLOOR
from shared.utils import get_logger, ensure_dir

logger = get_logger(__name__)

Q_TABLE_FILE = os.path.join(MODEL_SAVE_PATH, "rl_q_table.joblib")

# Discretise health score into 4 buckets: Critical, Warning, Moderate, Healthy
HEALTH_BUCKETS = [0, 30, 50, 80, 100]

# Actions per substation: reduce (-5%), hold (0%), increase (+5%)
ACTIONS = [-5.0, 0.0, 5.0]


def discretise_health(score: float) -> int:
    """Map health score to bucket index 0–3."""
    for i, threshold in enumerate(HEALTH_BUCKETS[1:]):
        if score <= threshold:
            return i
    return len(HEALTH_BUCKETS) - 2


class ReinforcementAgent:
    """
    Tabular Q-Learning agent for load balancing.

    State space:  (health_bucket_s1, health_bucket_s2, ...) — discretised
    Action space: combination of load adjustments per substation
    """

    def __init__(
        self,
        n_substations: int = 3,
        learning_rate: float = 0.1,
        discount:      float = 0.9,
        epsilon:       float = 0.2,   # exploration rate
    ):
        self.n_substations = n_substations
        self.lr       = learning_rate
        self.gamma    = discount
        self.epsilon  = epsilon

        # Q-table: state → action → Q-value
        # State is a tuple of health buckets, action is index into action combinations
        self._q_table: Dict[tuple, np.ndarray] = {}
        self._action_combos = self._build_action_combos()
        self._episode_rewards: List[float] = []

    # ── Public ────────────────────────────────────────────────────────────────

    def choose_action(self, health_data: Dict[str, dict]) -> Dict[str, float]:
        """
        Choose load adjustments for each substation.

        Returns: {sub_id: adjustment_pct}  e.g. {'S1': 5.0, 'S2': -5.0, 'S3': 0.0}
        """
        state = self._encode_state(health_data)
        substations = sorted(health_data.keys())

        if np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(len(self._action_combos))
        else:
            # Exploit: best known action
            q_values = self._get_q_values(state)
            action_idx = int(np.argmax(q_values))

        adjustments = self._action_combos[action_idx]
        return {sub: float(adj) for sub, adj in zip(substations, adjustments)}

    def update(
        self,
        health_before: Dict[str, dict],
        action:        Dict[str, float],
        health_after:  Dict[str, dict],
    ) -> float:
        """
        Update Q-table based on observed transition.
        Returns the TD error (useful for monitoring learning).
        """
        state      = self._encode_state(health_before)
        next_state = self._encode_state(health_after)
        reward     = self._compute_reward(health_before, health_after)

        substations  = sorted(health_before.keys())
        action_combo = tuple(action.get(s, 0.0) for s in substations)
        action_idx   = self._find_action_idx(action_combo)

        # Q-learning update: Q(s,a) += lr * (r + γ·max Q(s',a') - Q(s,a))
        q_values      = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)
        td_target     = reward + self.gamma * float(np.max(next_q_values))
        td_error      = td_target - q_values[action_idx]
        q_values[action_idx] += self.lr * td_error
        self._q_table[state] = q_values

        self._episode_rewards.append(reward)
        return float(td_error)

    def get_stats(self) -> dict:
        """Return learning statistics."""
        if not self._episode_rewards:
            return {"episodes": 0, "avg_reward": 0.0, "q_table_size": 0}
        recent = self._episode_rewards[-100:]
        return {
            "episodes":      len(self._episode_rewards),
            "avg_reward":    round(float(np.mean(recent)), 3),
            "max_reward":    round(float(np.max(recent)), 3),
            "q_table_size":  len(self._q_table),
            "epsilon":       self.epsilon,
        }

    def decay_epsilon(self, min_epsilon: float = 0.05, decay: float = 0.995) -> None:
        """Reduce exploration rate over time."""
        self.epsilon = max(min_epsilon, self.epsilon * decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        ensure_dir(MODEL_SAVE_PATH)
        joblib.dump({
            "q_table":         self._q_table,
            "episode_rewards": self._episode_rewards,
            "epsilon":         self.epsilon,
        }, Q_TABLE_FILE)
        logger.info(f"[RL Agent] Q-table saved ({len(self._q_table)} states).")

    def load(self) -> bool:
        if os.path.exists(Q_TABLE_FILE):
            data = joblib.load(Q_TABLE_FILE)
            self._q_table        = data["q_table"]
            self._episode_rewards = data["episode_rewards"]
            self.epsilon         = data["epsilon"]
            logger.info(f"[RL Agent] Loaded Q-table ({len(self._q_table)} states).")
            return True
        logger.info("[RL Agent] No saved Q-table — starting fresh.")
        return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _encode_state(self, health_data: Dict[str, dict]) -> tuple:
        substations = sorted(health_data.keys())
        return tuple(
            discretise_health(health_data[s].get("health_score", 50))
            for s in substations
        )

    def _get_q_values(self, state: tuple) -> np.ndarray:
        if state not in self._q_table:
            self._q_table[state] = np.zeros(len(self._action_combos))
        return self._q_table[state]

    def _compute_reward(
        self,
        health_before: Dict[str, dict],
        health_after:  Dict[str, dict],
    ) -> float:
        """Reward = improvement in average health score."""
        def avg_health(h):
            scores = [v.get("health_score", 50) for v in h.values()]
            return float(np.mean(scores)) if scores else 50.0

        before = avg_health(health_before)
        after  = avg_health(health_after)
        reward = after - before

        # Bonus for eliminating critical substations
        critical_before = sum(1 for v in health_before.values() if v.get("risk_level") == "Critical")
        critical_after  = sum(1 for v in health_after.values()  if v.get("risk_level") == "Critical")
        reward += (critical_before - critical_after) * 10.0

        return float(reward)

    def _build_action_combos(self) -> List[tuple]:
        """Build all combinations of actions for n_substations."""
        from itertools import product
        return list(product(ACTIONS, repeat=self.n_substations))

    def _find_action_idx(self, action_combo: tuple) -> int:
        try:
            return self._action_combos.index(action_combo)
        except ValueError:
            return 0
