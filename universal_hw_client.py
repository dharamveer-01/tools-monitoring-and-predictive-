"""
UniversalHardwareClient — auto-detects any connected hardware and streams
real telemetry to the AI server.

Detection order:
  1. Android phone via ADB (USB debugging enabled)
  2. Any COM/serial device (Arduino, ESP32, Pico, etc.)

If NO hardware is found, the client waits and prints instructions.
It NEVER generates fake/synthetic data.

Usage:
  python substations/universal_hw_client.py --id S1 --host 192.168.1.100
"""
import json
import time
import socket
import os
import subprocess
import argparse
import sys
from datetime import datetime

import serial
import serial.tools.list_ports

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import SERVER_PORT
from shared.utils import get_logger

logger = get_logger(__name__)


class UniversalHardwareClient:
    def __init__(
        self,
        substation_id: str = "S_HW",
        host: str = "localhost",
        server_port: int = SERVER_PORT,
    ):
        self.substation_id = substation_id
        self.host = host
        self.server_port = server_port

        self._socket: socket.socket | None = None
        self._serial: serial.Serial | None = None
        self._adb_cmd = self._find_adb()

    # ── Public ────────────────────────────────────────────────────────────────

    def start_streaming(self) -> None:
        self._connect_server()
        logger.info(f"[{self.substation_id}] Scanning for hardware…")

        try:
            while True:
                packet = self._read_hardware()
                if packet:
                    self._send(packet)
                    time.sleep(1.5)
                else:
                    self._print_no_hardware_help()
                    time.sleep(5)
        except KeyboardInterrupt:
            logger.info(f"[{self.substation_id}] Stopped.")
        finally:
            self._cleanup()

    # ── Hardware detection ────────────────────────────────────────────────────

    def _read_hardware(self) -> dict | None:
        """Try ADB first, then serial. Returns a packet or None."""
        packet = self._try_adb()
        if packet:
            return packet
        return self._try_serial()

    def _try_adb(self) -> dict | None:
        """Read real data from an Android phone via ADB."""
        if not self._adb_cmd:
            return None
        try:
            out = subprocess.run(
                [self._adb_cmd, "devices"],
                capture_output=True, text=True, timeout=3,
            ).stdout
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            # lines[0] = "List of devices attached", lines[1+] = device entries
            device_lines = [l for l in lines[1:] if "\tdevice" in l and "unauthorized" not in l]
            if not device_lines:
                return None

            battery = subprocess.run(
                [self._adb_cmd, "shell", "dumpsys", "battery"],
                capture_output=True, text=True, timeout=5,
            ).stdout

            voltage, temp, load = 0.0, 0.0, 0.0
            found_voltage = found_temp = found_load = False

            for line in battery.splitlines():
                line = line.strip()
                if line.startswith("level:"):
                    load = float(line.split(":")[1].strip())
                    found_load = True
                elif line.startswith("temperature:"):
                    val = float(line.split(":")[1].strip())
                    temp = val / 10.0 if val > 100 else val
                    found_temp = True
                elif line.startswith("voltage:"):
                    val = float(line.split(":")[1].strip())
                    voltage = val / 1000.0 if val > 1000 else val
                    found_voltage = True

            if not (found_voltage or found_temp or found_load):
                return None

            logger.info(f"[{self.substation_id}] ADB: V={voltage}V T={temp}°C Load={load}%")
            return {
                "substation_id":  self.substation_id,
                "timestamp":      datetime.now().isoformat(),
                "voltage":        round(voltage, 3),
                "current":        None,   # phone doesn't expose this
                "temperature":    round(temp, 2),
                "harmonic_5th":   None,   # phone doesn't expose this
                "load_percentage": round(load, 2),
            }
        except Exception as e:
            logger.debug(f"ADB read failed: {e}")
            return None

    def _try_serial(self) -> dict | None:
        """Read a JSON line from the first available COM port."""
        try:
            # Open port if not already open
            if not self._serial or not self._serial.is_open:
                ports = list(serial.tools.list_ports.comports())
                if not ports:
                    return None
                port_name = ports[0].device
                self._serial = serial.Serial(port_name, 9600, timeout=1)
                logger.info(f"[{self.substation_id}] Serial opened: {port_name}")

            if self._serial.in_waiting == 0:
                return None

            raw = self._serial.readline()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("{"):
                return None

            data = json.loads(line)
            packet = {
                "substation_id":  data.get("substation_id", self.substation_id),
                "timestamp":      datetime.now().isoformat(),
                "voltage":        float(data["voltage"])        if "voltage"         in data else None,
                "current":        float(data["current"])        if "current"         in data else None,
                "temperature":    float(data["temperature"])    if "temperature"     in data else None,
                "harmonic_5th":   float(data["harmonic_5th"])   if "harmonic_5th"   in data else None,
                "load_percentage": float(data["load_percentage"]) if "load_percentage" in data else None,
            }
            logger.info(f"[{self.substation_id}] Serial: {packet}")
            return packet

        except json.JSONDecodeError as e:
            logger.warning(f"[{self.substation_id}] Bad JSON from serial: {e}")
            return None
        except serial.SerialException as e:
            logger.warning(f"[{self.substation_id}] Serial error: {e}")
            if self._serial:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None
            return None

    # ── Server connection ─────────────────────────────────────────────────────

    def _connect_server(self) -> None:
        while True:
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.server_port))
                logger.info(f"[{self.substation_id}] Connected to AI server {self.host}:{self.server_port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(f"[{self.substation_id}] Server unreachable: {e}. Retrying in 2s…")
                time.sleep(2)

    def _send(self, packet: dict) -> None:
        message = json.dumps(packet) + "\n"
        try:
            self._socket.sendall(message.encode("utf-8"))
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"[{self.substation_id}] Send failed: {e} — reconnecting…")
            try:
                self._socket.close()
            except Exception:
                pass
            self._connect_server()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_adb(self) -> str | None:
        local = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "platform-tools", "adb.exe",
        )
        if os.path.exists(local):
            return local
        try:
            subprocess.run(["adb", "version"], capture_output=True, timeout=2)
            return "adb"
        except Exception:
            return None

    def _print_no_hardware_help(self) -> None:
        logger.warning(
            f"[{self.substation_id}] No hardware detected. "
            "Plug in a USB sensor (Arduino/ESP32) or an Android phone with USB debugging enabled. "
            "Retrying in 5 seconds…"
        )

    def _cleanup(self) -> None:
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass
        try:
            if self._socket:
                self._socket.close()
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal hardware client (ADB or serial)")
    parser.add_argument("--id",          default="S_HW",      help="Substation ID")
    parser.add_argument("--host",        default="localhost",  help="AI server IP")
    parser.add_argument("--server-port", default=SERVER_PORT,  type=int)
    args = parser.parse_args()

    client = UniversalHardwareClient(
        substation_id=args.id,
        host=args.host,
        server_port=args.server_port,
    )
    client.start_streaming()


if __name__ == "__main__":
    main()
