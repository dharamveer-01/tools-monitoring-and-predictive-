"""
SmartGridSocketServer — TCP server that receives telemetry from all substations
and runs the full AI pipeline on every packet.

AI Pipeline per packet:
  1.  Validate & store telemetry
  2.  Feature engineering
  3.  IsolationForest anomaly detection  (point-based, instant)
  4.  LSTM sequence anomaly detection    (trend-based, after 10 readings)
  5.  Health score calculation
  6.  Fault isolation
  7.  Overload prediction                (after 10 readings)
  8.  Transformer failure prediction     (after 20 readings)
  9.  Alert generation (with cooldown)
  10. Smart load redistribution          (via RedistributionEngine)
  11. Self-healing notification
"""
import socket
import threading
import sys
import os
from datetime import datetime, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.anomaly_detector import AnomalyDetector
from ml.health_score import calculate_health_score
from smart_grid.load_balancer import LoadBalancer
from smart_grid.fault_isolation import isolate_faults
from smart_grid.self_healing_engine import SelfHealingEngine
from alerts.alert_manager import AlertManager
from server.connection_handler import ConnectionHandler
from server.telemetry_manager import TelemetryManager
from shared.utils import get_logger
from shared.config import SERVER_HOST, SERVER_PORT, ALERT_COOLDOWN_SEC

# Max concurrent substation connections
MAX_CONNECTIONS = 50

# ── Advanced ML models (wired in from ml_models/) ────────────────────────────
try:
    from ml_models.anomaly_detection.lstm_anomaly_detector import LSTMAnomalyDetector
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTM_AVAILABLE = False

try:
    from ml_models.prediction.overload_predictor import OverloadPredictor
    _OVERLOAD_AVAILABLE = True
except ImportError:
    _OVERLOAD_AVAILABLE = False

try:
    from ml_models.prediction.transformer_failure_predictor import TransformerFailurePredictor
    _FAILURE_AVAILABLE = True
except ImportError:
    _FAILURE_AVAILABLE = False

try:
    from ml_models.load_balancing.redistribution_engine import RedistributionEngine
    _REDIST_AVAILABLE = True
except ImportError:
    _REDIST_AVAILABLE = False

logger = get_logger(__name__)


class SmartGridSocketServer:
    def __init__(self, host: str = SERVER_HOST, port: int = SERVER_PORT):
        self.host = host
        self.port = port

        # ── State stores ──────────────────────────────────────────────────────
        self.telemetry_data: dict = {}
        self.health_data:    dict = {}
        self.fault_reports:  dict = {}
        self.predictions:    dict = {}   # sub_id → {overload, failure}

        # ── Core components ───────────────────────────────────────────────────
        self.telemetry_manager = TelemetryManager()
        self.anomaly_detector  = AnomalyDetector()
        self.balancer          = LoadBalancer()
        self.alert_manager     = AlertManager()
        self.healer            = SelfHealingEngine(self.balancer)

        # ── Advanced ML components ────────────────────────────────────────────
        # LSTM detectors: one per substation (keyed by sub_id)
        self._lstm_detectors: dict = {}

        # Overload predictor (shared, stateless per call)
        self.overload_predictor = None
        if _OVERLOAD_AVAILABLE:
            self.overload_predictor = OverloadPredictor()
            if not self.overload_predictor.load():
                logger.info("[Server] Training OverloadPredictor…")
                self.overload_predictor.train()
                self.overload_predictor.save()

        # Transformer failure predictor (shared, stateless per call)
        self.failure_predictor = None
        if _FAILURE_AVAILABLE:
            self.failure_predictor = TransformerFailurePredictor()
            if not self.failure_predictor.load():
                logger.info("[Server] Training TransformerFailurePredictor…")
                self.failure_predictor.train()
                self.failure_predictor.save()

        # Smart redistribution engine (replaces simple balancer calls)
        self.redistribution_engine = None
        if _REDIST_AVAILABLE:
            self.redistribution_engine = RedistributionEngine(self.balancer)
            logger.info("[Server] RedistributionEngine active.")

        # Alert cooldown tracking
        self._last_alert: dict = {}
        self._lock = threading.Lock()
        self._connection_semaphore = threading.Semaphore(MAX_CONNECTIONS)

        # Socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        logger.info(
            f"[Server] Advanced ML: "
            f"LSTM={'✓' if _LSTM_AVAILABLE else '✗'}  "
            f"Overload={'✓' if _OVERLOAD_AVAILABLE else '✗'}  "
            f"Failure={'✓' if _FAILURE_AVAILABLE else '✗'}  "
            f"SmartRedist={'✓' if _REDIST_AVAILABLE else '✗'}"
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self, ready_event: threading.Event | None = None) -> None:
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)
        self.healer.start()
        logger.info(f"[SERVER] Listening on {self.host}:{self.port}")
        if ready_event:
            ready_event.set()   # signal that the server is ready to accept
        try:
            while True:
                conn, addr = self._server_socket.accept()
                if not self._connection_semaphore.acquire(blocking=False):
                    logger.warning(f"[SERVER] Max connections ({MAX_CONNECTIONS}) reached. Rejecting {addr}")
                    conn.close()
                    continue
                handler = ConnectionHandler(
                    conn=conn,
                    addr=addr,
                    process_callback=self.process_telemetry,
                    disconnect_callback=self._on_disconnect,
                )
                t = threading.Thread(target=self._handle_with_semaphore, args=(handler,), daemon=True)
                t.start()
                logger.info(f"[CONNECTIONS] Active: {threading.active_count() - 1}")
        except KeyboardInterrupt:
            logger.info("Server shutting down.")
        finally:
            self._server_socket.close()

    def _handle_with_semaphore(self, handler) -> None:
        """Run handler and release semaphore when done."""
        try:
            handler.handle()
        finally:
            self._connection_semaphore.release()

    def process_telemetry(self, packet: dict) -> None:
        """Full AI pipeline for one telemetry packet."""
        sub_id = packet.get("substation_id")
        if not sub_id:
            return

        # Coerce numeric fields to float — prevents crashes on string values
        for field in ("voltage", "current", "temperature", "harmonic_5th", "load_percentage"):
            val = packet.get(field)
            if val is not None:
                try:
                    packet[field] = float(val)
                except (TypeError, ValueError):
                    packet[field] = None

        # 1. Store
        with self._lock:
            self.telemetry_data[sub_id] = packet
        self.telemetry_manager.update(packet)

        # 2. IsolationForest anomaly detection (point-based)
        anomaly_result = self.anomaly_detector.detect_with_score(packet)
        is_anomaly     = anomaly_result["anomaly"]

        # 3. LSTM sequence anomaly detection (trend-based)
        lstm_anomaly = False
        if _LSTM_AVAILABLE:
            lstm_anomaly = self._run_lstm_detection(sub_id, packet)
            # Combine: anomaly if either model flags it
            if lstm_anomaly and not is_anomaly:
                is_anomaly = True
                logger.info(f"[{sub_id}] LSTM detected sequence anomaly (IF missed it)")

        # 4. Health score
        score, status = calculate_health_score(packet, is_anomaly)
        previous_status = self.health_data.get(sub_id, {}).get("risk_level")

        health_record = {
            "anomaly_detected":    bool(is_anomaly),
            "anomaly_score":       float(anomaly_result.get("score", 0.0)),
            "lstm_anomaly":        bool(lstm_anomaly),
            "health_score":        float(score),
            "risk_level":          status,
            "timestamp":           packet.get("timestamp", datetime.now().isoformat()),
        }
        with self._lock:
            self.health_data[sub_id] = health_record

        # 5. Fault isolation
        fault_report = isolate_faults(packet)
        with self._lock:
            self.fault_reports[sub_id] = fault_report

        # 6. Overload + failure predictions (use rolling history)
        history = self.telemetry_manager.get_history(sub_id)
        pred_record = {}

        if self.overload_predictor and len(history) >= 5:
            pred_record["overload"] = self.overload_predictor.predict(history)

        if self.failure_predictor and len(history) >= 10:
            pred_record["transformer_failure"] = self.failure_predictor.predict(history)

        if pred_record:
            with self._lock:
                self.predictions[sub_id] = pred_record
            # Alert on high overload probability
            overload = pred_record.get("overload", {})
            if overload.get("will_overload") and overload.get("overload_probability", 0) > 0.75:
                self.alert_manager.trigger_alert(
                    sub_id,
                    f"Overload predicted in next {overload.get('horizon', 5)} readings. "
                    f"Probability: {overload['overload_probability']:.0%}",
                    risk_level="WARNING",
                )

        # 7. Alerts
        self._maybe_alert(sub_id, status, score, previous_status, fault_report)

        # 8. Load redistribution
        if self.redistribution_engine:
            self.redistribution_engine.check_and_redistribute(self.health_data)
        else:
            active_subs = list(self.telemetry_data.keys())
            self.balancer.redistribute(active_subs, self.health_data)

        # 9. Self-healing
        self.healer.notify_health(sub_id, score, status)

        logger.info(
            f"[{sub_id}] V={packet.get('voltage')}V "
            f"T={packet.get('temperature')}°C "
            f"Load={packet.get('load_percentage')}% "
            f"Score={score} ({status})"
            + (" ⚠IF" if anomaly_result["anomaly"] else "")
            + (" ⚠LSTM" if lstm_anomaly else "")
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_lstm_detection(self, sub_id: str, packet: dict) -> bool:
        """Run LSTM sequence detector for this substation."""
        if sub_id not in self._lstm_detectors:
            detector = LSTMAnomalyDetector()
            if not detector.load():
                detector.train()
                detector.save()
            self._lstm_detectors[sub_id] = detector

        detector = self._lstm_detectors[sub_id]
        detector.add_reading(packet)
        result = detector.predict_current()
        return result.get("anomaly", False)

    def _maybe_alert(self, sub_id, status, score, previous_status, fault_report) -> None:
        now = datetime.now()
        last = self._last_alert.get(sub_id)
        cooldown_ok = (last is None) or (now - last > timedelta(seconds=ALERT_COOLDOWN_SEC))

        if status == "Critical" and cooldown_ok:
            faults = ", ".join(f["name"] for f in fault_report.get("faults_detected", []))
            msg = f"Critical state. Score: {score}."
            if faults:
                msg += f" Faults: {faults}."
            self.alert_manager.trigger_alert(sub_id, msg, risk_level="CRITICAL")
            self._last_alert[sub_id] = now

        elif status == "Warning" and previous_status == "Healthy" and cooldown_ok:
            self.alert_manager.trigger_alert(
                sub_id, f"Health degraded to Warning. Score: {score}.", risk_level="WARNING"
            )
            self._last_alert[sub_id] = now

        elif status == "Healthy" and previous_status in ("Critical", "Warning"):
            self.alert_manager.trigger_alert(
                sub_id, f"Substation recovered. Score: {score}.", risk_level="INFO"
            )

    def _on_disconnect(self, sub_id: str) -> None:
        with self._lock:
            self.telemetry_data.pop(sub_id, None)
            self.health_data.pop(sub_id, None)
            self.fault_reports.pop(sub_id, None)
            self.predictions.pop(sub_id, None)
            self._lstm_detectors.pop(sub_id, None)
        self.telemetry_manager.remove(sub_id)
        logger.info(f"[CLEANUP] Removed state for {sub_id}")


if __name__ == "__main__":
    server = SmartGridSocketServer()
    server.start()
