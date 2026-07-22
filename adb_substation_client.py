"""
AdbSubstationClient — reads REAL data from an Android phone via USB (ADB).

What the phone actually provides:
  battery level    → load_percentage  (0–100%)       REAL
  battery temp     → temperature      (°C)            REAL
  battery voltage  → scaled to grid   (220–240V)      REAL (scaled)
  cpu usage        → current          (10–20A range)  REAL (mapped)
  charging status  → harmonic_5th     (fault proxy)   REAL (mapped)

Voltage mapping:
  Phone battery range: 3.3V (empty) → 4.2V (full)
  Grid equivalent:     200V          → 245V
  Formula: grid_v = 200 + ((batt_v - 3.3) / (4.2 - 3.3)) * 45

Usage:
  python substations/substation_client.py --id S1 --source adb
  python substations/adb_substation_client.py
"""
import json
import time
import socket
import os
import subprocess
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import SERVER_PORT
from shared.utils import get_logger

logger = get_logger(__name__)

# ── Voltage scaling constants ──────────────────────────────────────────────────
BATT_V_MIN  = 3.3    # phone battery empty
BATT_V_MAX  = 4.2    # phone battery full
GRID_V_MIN  = 200.0  # grid voltage at low battery
GRID_V_MAX  = 245.0  # grid voltage at full battery


def battery_voltage_to_grid(batt_v: float) -> float:
    """Map phone battery voltage (3.3–4.2V) to grid voltage (200–245V)."""
    ratio = (batt_v - BATT_V_MIN) / (BATT_V_MAX - BATT_V_MIN)
    ratio = max(0.0, min(1.0, ratio))
    return round(GRID_V_MIN + ratio * (GRID_V_MAX - GRID_V_MIN), 2)


def battery_level_to_current(level: float) -> float:
    """
    Map battery level (0–100%) to simulated current (10–20A).
    Higher battery = more stable = higher current draw (healthy substation).
    """
    return round(10.0 + (level / 100.0) * 10.0, 2)


def charging_to_harmonic(charging: bool, plugged: str) -> float:
    """
    Map charging state to harmonic distortion proxy.
    Charging via AC = clean power = low harmonics.
    Discharging / USB = higher harmonic proxy.
    """
    if charging and plugged in ("AC", "Wireless"):
        return 1.5   # clean
    elif charging:
        return 3.0   # USB charging — slight distortion
    else:
        return 5.5   # discharging — higher distortion proxy


class AdbSubstationClient:
    def __init__(self, substation_id: str = "S1", host: str = "localhost", port: int = SERVER_PORT):
        self.substation_id = substation_id
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._adb_cmd = self._find_adb()
        self._device_connected = False

    # ── ADB discovery ─────────────────────────────────────────────────────────

    def _find_adb(self) -> str | None:
        local = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "platform-tools", "adb.exe",
        )
        if os.path.exists(local):
            logger.info(f"Using local ADB: {local}")
            self._restart_adb_server(local)
            return local
        try:
            subprocess.run(["adb", "version"], capture_output=True, timeout=3)
            self._restart_adb_server("adb")
            return "adb"
        except Exception:
            logger.error(
                "ADB not found. Run: python setup_adb.py\n"
                "Or download from: https://developer.android.com/tools/releases/platform-tools"
            )
            return None

    def _restart_adb_server(self, adb_cmd: str) -> None:
        """Kill and restart the ADB server to clear any stale state."""
        try:
            logger.info("[ADB] Restarting ADB server to clear stale connections…")
            subprocess.run([adb_cmd, "kill-server"], capture_output=True, timeout=5)
            time.sleep(1)
            subprocess.run([adb_cmd, "start-server"], capture_output=True, timeout=10)
            time.sleep(1)
            logger.info("[ADB] ADB server restarted.")
        except Exception as e:
            logger.warning(f"[ADB] Could not restart server: {e}")

    # ── Public ────────────────────────────────────────────────────────────────

    def start_streaming(self) -> None:
        if not self._adb_cmd:
            logger.error("Cannot start — ADB not available.")
            return

        self._connect_server()
        logger.info(f"[{self.substation_id}] ADB client started. Waiting for phone…")

        try:
            while True:
                packet = self._read_phone()
                if packet:
                    self._send(packet)
                else:
                    time.sleep(2)
        except KeyboardInterrupt:
            logger.info(f"[{self.substation_id}] Stopped.")
        finally:
            if self._socket:
                self._socket.close()

    # ── Phone data extraction ─────────────────────────────────────────────────

    def _read_phone(self) -> dict | None:
        """Extract real telemetry from the connected Android phone via ADB."""
        try:
            # Check device is connected and authorised
            out = subprocess.run(
                [self._adb_cmd, "devices"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            device_lines = [
                l for l in out.splitlines()
                if "\tdevice" in l and "unauthorized" not in l
            ]
            if not device_lines:
                # Check for unauthorized (USB debugging not accepted yet)
                unauth = [l for l in out.splitlines() if "unauthorized" in l]
                if unauth:
                    logger.warning(
                        f"[{self.substation_id}] Phone found but UNAUTHORIZED.\n"
                        "  → On your phone: tap 'Allow' on the USB Debugging popup\n"
                        "  → If no popup appears: unplug and replug the USB cable"
                    )
                else:
                    if self._device_connected:
                        logger.warning(f"[{self.substation_id}] Phone disconnected.")
                        self._device_connected = False
                    else:
                        logger.info(
                            f"[{self.substation_id}] Waiting for Android phone via USB…\n"
                            "  → Enable USB Debugging: Settings → Developer Options → USB Debugging\n"
                            "  → Then accept the 'Allow USB Debugging' popup on your phone"
                        )
                return None

            if not self._device_connected:
                logger.info(f"[{self.substation_id}] Phone connected: {device_lines[0].split()[0]}")
                self._device_connected = True

            # ── Battery data ──────────────────────────────────────────────────
            battery_raw = subprocess.run(
                [self._adb_cmd, "shell", "dumpsys", "battery"],
                capture_output=True, text=True, timeout=5,
            ).stdout

            batt_v_raw    = None
            batt_temp_raw = None
            batt_level    = 50.0
            charging      = False
            plugged_type  = "none"

            for line in battery_raw.splitlines():
                line = line.strip()
                if line.startswith("level:"):
                    batt_level = float(line.split(":")[1].strip())
                elif line.startswith("temperature:"):
                    batt_temp_raw = float(line.split(":")[1].strip())
                elif line.startswith("voltage:"):
                    batt_v_raw = float(line.split(":")[1].strip())
                elif line.startswith("status:"):
                    # 2 = charging, 3 = discharging, 4 = not charging, 5 = full
                    status_val = line.split(":")[1].strip()
                    charging = status_val in ("2", "5")
                elif line.startswith("plugged:"):
                    plugged_val = line.split(":")[1].strip()
                    plugged_type = {"1": "AC", "2": "USB", "4": "Wireless"}.get(plugged_val, "none")

            # ── Convert raw values ────────────────────────────────────────────
            # Battery voltage: Android reports in mV (e.g. 3805 → 3.805V)
            if batt_v_raw is not None:
                batt_v = batt_v_raw / 1000.0 if batt_v_raw > 100 else batt_v_raw
            else:
                batt_v = 3.7  # fallback

            # Battery temperature: Android reports in decidegrees (e.g. 346 → 34.6°C)
            if batt_temp_raw is not None:
                temp = batt_temp_raw / 10.0 if batt_temp_raw > 100 else batt_temp_raw
            else:
                temp = 35.0

            # ── Try to get CPU usage for current mapping ──────────────────────
            cpu_usage = self._get_cpu_usage()

            # ── Map to grid telemetry ─────────────────────────────────────────
            grid_voltage = battery_voltage_to_grid(batt_v)
            current      = round(10.0 + (cpu_usage / 100.0) * 10.0, 2)  # CPU → current
            harmonic     = charging_to_harmonic(charging, plugged_type)

            packet = {
                "substation_id":   self.substation_id,
                "timestamp":       datetime.now().isoformat(),
                "voltage":         grid_voltage,       # scaled from battery voltage
                "current":         current,            # mapped from CPU usage
                "temperature":     round(temp, 1),     # real battery temperature
                "harmonic_5th":    harmonic,           # mapped from charging state
                "load_percentage": round(batt_level, 1), # real battery level
                # Extra phone metadata (shown in dashboard, not used by AI)
                "_source":         "android_adb",
                "_battery_v":      round(batt_v, 3),
                "_battery_pct":    batt_level,
                "_charging":       charging,
                "_plugged":        plugged_type,
                "_cpu_pct":        cpu_usage,
            }

            logger.info(
                f"[{self.substation_id}] Phone → "
                f"BattV={batt_v:.3f}V→GridV={grid_voltage}V | "
                f"Temp={temp:.1f}°C | "
                f"Batt={batt_level:.0f}% | "
                f"CPU={cpu_usage:.0f}% | "
                f"{'⚡ Charging' if charging else '🔋 Discharging'} ({plugged_type})"
            )
            return packet

        except subprocess.TimeoutExpired:
            logger.warning(f"[{self.substation_id}] ADB command timed out — restarting ADB server…")
            self._restart_adb_server(self._adb_cmd)
            return None
        except Exception as e:
            logger.error(f"[{self.substation_id}] ADB read error: {e}")
            return None

    def _get_cpu_usage(self) -> float:
        """Get overall CPU usage % from the phone. Returns 0–100."""
        try:
            # Try top -n 1 -d 1 and parse the CPU line
            out = subprocess.run(
                [self._adb_cmd, "shell", "top", "-n", "1", "-d", "1"],
                capture_output=True, text=True, timeout=6,
            ).stdout
            for line in out.splitlines():
                line_lower = line.lower()
                if "cpu" in line_lower and "%" in line:
                    # Format: "%cpu  X% user  Y% sys  Z% idle ..."
                    # or: "800%cpu  12%user  0%nice  15%sys  773%idle ..."
                    import re
                    idle_match = re.search(r'(\d+)%idle', line_lower)
                    if idle_match:
                        idle = float(idle_match.group(1))
                        # Normalise: top may report total across all cores
                        # Cap at 100 for display
                        return min(100.0, max(0.0, 100.0 - idle))
            return 30.0  # fallback if parsing fails
        except Exception:
            return 30.0  # safe fallback

    # ── Server connection ─────────────────────────────────────────────────────

    def _connect_server(self) -> None:
        while True:
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.port))
                logger.info(f"[{self.substation_id}] Connected to AI server {self.host}:{self.port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(f"[{self.substation_id}] Server unreachable: {e}. Retrying in 2s…")
                time.sleep(2)

    def _send(self, packet: dict) -> None:
        # Strip private metadata fields before sending to server
        # (server only needs the standard telemetry fields)
        clean = {k: v for k, v in packet.items() if not k.startswith("_")}
        message = json.dumps(clean) + "\n"
        try:
            self._socket.sendall(message.encode("utf-8"))
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"[{self.substation_id}] Send failed: {e} — reconnecting…")
            try:
                self._socket.close()
            except Exception:
                pass
            self._connect_server()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ADB Android phone substation client")
    parser.add_argument("--id",   default="S1",       help="Substation ID")
    parser.add_argument("--host", default="localhost", help="AI server IP")
    parser.add_argument("--port", default=SERVER_PORT, type=int)
    args = parser.parse_args()

    client = AdbSubstationClient(
        substation_id=args.id,
        host=args.host,
        port=args.port,
    )
    client.start_streaming()
