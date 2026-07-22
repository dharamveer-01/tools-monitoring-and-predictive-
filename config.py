"""
Centralized configuration for the Distributed AI Smart Grid Simulator.
All tuneable parameters live here — change once, applies everywhere.
"""

# ─── Network ──────────────────────────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"          # Bind address for the AI server
SERVER_PORT = 9999                # TCP socket port for telemetry streaming
API_HOST    = "0.0.0.0"
API_PORT    = 8000

# ─── Substation defaults ──────────────────────────────────────────────────────
DEFAULT_SUBSTATIONS = ["S1", "S2", "S3"]
TELEMETRY_INTERVAL  = 1.5        # seconds between telemetry packets

# ─── Normal operating ranges ──────────────────────────────────────────────────
VOLTAGE_MIN   = 220.0
VOLTAGE_MAX   = 240.0
CURRENT_MIN   = 10.0
CURRENT_MAX   = 20.0
TEMP_MIN      = 50.0
TEMP_MAX      = 75.0
HARMONIC_MIN  = 1.0
HARMONIC_MAX  = 5.0
LOAD_MIN      = 20.0
LOAD_MAX      = 60.0

# ─── Fault / anomaly thresholds ───────────────────────────────────────────────
FAULT_VOLTAGE_LOW   = 170.0
FAULT_VOLTAGE_HIGH  = 190.0
FAULT_TEMP_HIGH     = 100.0
FAULT_HARMONIC_HIGH = 10.0
FAULT_LOAD_HIGH     = 85.0

# ─── Health score thresholds ──────────────────────────────────────────────────
HEALTH_HEALTHY_MIN  = 80
HEALTH_WARNING_MIN  = 50
# Below HEALTH_WARNING_MIN → Critical

# ─── Load balancing ───────────────────────────────────────────────────────────
CRITICAL_LOAD_FLOOR = 10.0       # Minimum load assigned to a critical substation
DEFAULT_LOAD_SHARE  = 33.3       # Equal share when all substations are healthy

# ─── Self-healing ─────────────────────────────────────────────────────────────
RECOVERY_THRESHOLD  = 70         # Health score above which recovery begins
RECOVERY_STEP       = 5.0        # Load % restored per healing cycle
HEALING_INTERVAL    = 10         # seconds between healing checks

# ─── Alerts ───────────────────────────────────────────────────────────────────
MAX_ALERT_HISTORY   = 100
ALERT_COOLDOWN_SEC  = 30         # Minimum seconds between same-substation alerts

# ─── ML model ─────────────────────────────────────────────────────────────────
ISOLATION_FOREST_CONTAMINATION = 0.1
ISOLATION_FOREST_TRAINING_SAMPLES = 500
MODEL_SAVE_PATH = "ml_models/saved_models"

# ─── Feature engineering ──────────────────────────────────────────────────────
HISTORY_WINDOW = 20              # Number of past readings kept per substation
