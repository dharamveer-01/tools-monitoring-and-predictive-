"""
UsbSubstationClient — reads REAL sensor data from a USB-connected device
(Arduino, ESP32, Raspberry Pi Pico, or any microcontroller) and streams
it to the AI server.

Expected serial protocol:
  The microcontroller must send newline-terminated JSON every ~1 second:

  {"voltage": 230.1, "current": 15.2, "temperature": 60.5,
   "harmonic_5th": 2.1, "load_percentage": 45.0}

  Any subset of fields is accepted — missing fields are flagged as None
  and the packet is still forwarded so the server can decide what to do.

Usage:
  # Auto-detect port:
  python substations/usb_substation_client.py --id S1 --host 192.168.1.100

  # Specify port explicitly:
  python substations/usb_substation_client.py --id S1 --host 192.168.1.100 --port-name COM3
  python substations/usb_substation_client.py --id S1 --host 192.168.1.100 --port-name /dev/ttyUSB0

  # List available ports and exit:
  python substations/usb_substation_client.py --list-ports
"""
import json
import time
import socket
import argparse
import sys
import os
from datetime import datetime

import serial
import serial.tools.list_ports

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import SERVER_PORT
from shared.utils import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = ["voltage", "current", "temperature", "harmonic_5th", "load_percentage"]


# ── Port discovery ─────────────────────────────────────────────────────────────

def list_ports() -> list:
    """Return all available COM/serial ports."""
    return list(serial.tools.list_ports.comports())


def auto_detect_port() -> str | None:
    """Return the first available COM port, or None if none found."""
    ports = list_ports()
    if ports:
        chosen = ports[0].device
        logger.info(f"Auto-detected USB port: {chosen} ({ports[0].description})")
        return chosen
    return None


def print_available_ports() -> None:
    ports = list_ports()
    if not ports:
        print("No USB/COM ports found. Check that your device is plugged in.")
        return
    print(f"Found {len(ports)} port(s):")
    for p in ports:
        print(f"  {p.device:12s}  {p.description}  [{p.hwid}]")


# ── Client ────────────────────────────────────────────────────────────────────

class UsbSubstationClient:
    def __init__(
        self,
        substation_id: str,
        host: str = "localhost",
        server_port: int = SERVER_PORT,
        port_name: str | None = None,
        baud_rate: int = 9600,
    ):
        self.substation_id = substation_id
        self.host = host
        self.server_port = server_port
        self.port_name = port_name      # None = auto-detect
        self.baud_rate = baud_rate

        self._serial: serial.Serial | None = None
        self._socket: socket.socket | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def start_streaming(self) -> None:
        """Open USB port, connect to server, stream forever."""
        self._open_serial()
        self._connect_server()

        logger.info(f"[{self.substation_id}] Streaming USB data → {self.host}:{self.server_port}")
        try:
            while True:
                packet = self._read_packet()
                if packet:
                    self._send(packet)
                else:
                    # No data yet — wait a bit before polling again
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info(f"[{self.substation_id}] Stopped by user.")
        finally:
            self._cleanup()

    # ── Serial ────────────────────────────────────────────────────────────────

    def _open_serial(self) -> None:
        """Open the serial port, retrying until a device is found."""
        while True:
            port = self.port_name or auto_detect_port()
            if port is None:
                logger.warning(
                    f"[{self.substation_id}] No USB device found. "
                    "Plug in your sensor and press Enter to retry, or Ctrl+C to quit."
                )
                try:
                    input()
                except EOFError:
                    time.sleep(3)
                continue

            try:
                self._serial = serial.Serial(port, self.baud_rate, timeout=2)
                logger.info(
                    f"[{self.substation_id}] USB serial opened: {port} @ {self.baud_rate} baud"
                )
                return
            except serial.SerialException as e:
                logger.error(f"[{self.substation_id}] Cannot open {port}: {e}")
                logger.info("Retrying in 3 seconds…")
                time.sleep(3)

    def _read_packet(self) -> dict | None:
        """
        Read one newline-terminated JSON line from the serial port.
        Returns a complete telemetry dict or None if no data is ready.
        """
        if not self._serial or not self._serial.is_open:
            logger.warning(f"[{self.substation_id}] Serial port closed — reopening…")
            self._open_serial()
            return None

        try:
            if self._serial.in_waiting == 0:
                return None

            raw = self._serial.readline()
            line = raw.decode("utf-8", errors="replace").strip()

            if not line:
                return None

            if not line.startswith("{"):
                # Not JSON — could be a debug print from the microcontroller
                logger.debug(f"[{self.substation_id}] Non-JSON from serial: {line}")
                return None

            sensor_data = json.loads(line)
            return self._build_packet(sensor_data)

        except json.JSONDecodeError as e:
            logger.warning(f"[{self.substation_id}] Bad JSON from serial: {line!r} — {e}")
            return None
        except serial.SerialException as e:
            logger.error(f"[{self.substation_id}] Serial read error: {e} — reconnecting…")
            self._close_serial()
            self._open_serial()
            return None

    def _build_packet(self, sensor_data: dict) -> dict:
        """
        Build a complete telemetry packet from raw sensor data.
        Fields present in sensor_data are used as-is.
        Missing fields are set to None so the server knows they are absent.
        """
        packet = {
            "substation_id": self.substation_id,
            "timestamp":     datetime.now().isoformat(),
        }
        for field in REQUIRED_FIELDS:
            raw_val = sensor_data.get(field)
            if raw_val is not None:
                try:
                    packet[field] = round(float(raw_val), 3)
                except (TypeError, ValueError):
                    logger.warning(
                        f"[{self.substation_id}] Field '{field}' has non-numeric value: {raw_val!r}"
                    )
                    packet[field] = None
            else:
                packet[field] = None

        # Log which fields are missing so the user knows what the sensor isn't sending
        missing = [f for f in REQUIRED_FIELDS if packet[f] is None]
        if missing:
            logger.warning(
                f"[{self.substation_id}] Sensor not sending: {missing}. "
                "Update your microcontroller firmware to include these fields."
            )

        return packet

    def _close_serial(self) -> None:
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass
        self._serial = None

    # ── TCP socket ────────────────────────────────────────────────────────────

    def _connect_server(self) -> None:
        """Connect to the AI server, retrying until successful."""
        while True:
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.server_port))
                logger.info(
                    f"[{self.substation_id}] Connected to AI server at {self.host}:{self.server_port}"
                )
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(
                    f"[{self.substation_id}] Cannot reach server at {self.host}:{self.server_port}: {e}. "
                    "Retrying in 2s…"
                )
                time.sleep(2)

    def _send(self, packet: dict) -> None:
        message = json.dumps(packet) + "\n"
        try:
            self._socket.sendall(message.encode("utf-8"))
            # Compact log line showing only non-None fields
            values = {k: v for k, v in packet.items() if k not in ("substation_id", "timestamp") and v is not None}
            logger.info(f"[{self.substation_id}] Sent: {values}")
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"[{self.substation_id}] Send failed: {e} — reconnecting to server…")
            try:
                self._socket.close()
            except Exception:
                pass
            self._connect_server()

    def _cleanup(self) -> None:
        self._close_serial()
        try:
            if self._socket:
                self._socket.close()
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="USB Substation Client — streams real sensor data to the AI server"
    )
    parser.add_argument("--id",          default="S1",       help="Substation ID (e.g. S1, S2, S3)")
    parser.add_argument("--host",        default="localhost", help="AI server IP address")
    parser.add_argument("--server-port", default=SERVER_PORT, type=int, help="AI server TCP port")
    parser.add_argument("--port-name",   default=None,        help="COM port (e.g. COM3, /dev/ttyUSB0). Auto-detects if omitted.")
    parser.add_argument("--baud",        default=9600,        type=int, help="Serial baud rate (default: 9600)")
    parser.add_argument("--list-ports",  action="store_true", help="List available USB ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        sys.exit(0)

    client = UsbSubstationClient(
        substation_id=args.id,
        host=args.host,
        server_port=args.server_port,
        port_name=args.port_name,
        baud_rate=args.baud,
    )
    client.start_streaming()


if __name__ == "__main__":
    main()
