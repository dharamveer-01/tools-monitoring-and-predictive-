"""
ConnectionHandler — manages a single TCP client connection.

Each connected substation gets its own ConnectionHandler instance running
in a dedicated thread.  It:
  1. Reads newline-delimited JSON from the socket
  2. Validates the packet
  3. Calls the provided process_callback(packet)
  4. Cleans up on disconnect
"""
import json
import socket
from typing import Callable

from shared.utils import get_logger, validate_telemetry

logger = get_logger(__name__)


class ConnectionHandler:
    def __init__(
        self,
        conn: socket.socket,
        addr: tuple,
        process_callback: Callable[[dict], None],
        disconnect_callback: Callable[[str], None] | None = None,
    ):
        self.conn = conn
        self.addr = addr
        self.process_callback = process_callback
        self.disconnect_callback = disconnect_callback
        self._sub_id: str | None = None

    def handle(self) -> None:
        """Blocking loop — call from a dedicated thread."""
        logger.info(f"New connection from {self.addr}")
        buffer = ""
        try:
            while True:
                chunk = self.conn.recv(4096).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buffer += chunk

                # Process all complete newline-terminated messages
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._handle_line(line)

        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            logger.warning(f"Connection {self.addr} dropped: {e}")
        except Exception as e:
            logger.error(f"Unexpected error from {self.addr}: {e}")
        finally:
            self._cleanup()

    # ── private ───────────────────────────────────────────────────────────────

    def _handle_line(self, line: str) -> None:
        try:
            packet = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Bad JSON from {self.addr}: {line[:80]}")
            return

        if not validate_telemetry(packet):
            logger.warning(f"Invalid telemetry from {self.addr}: {packet}")
            return

        # Track which substation this connection belongs to
        if self._sub_id is None:
            self._sub_id = packet.get("substation_id")
            logger.info(f"Substation {self._sub_id} identified at {self.addr}")

        self.process_callback(packet)

    def _cleanup(self) -> None:
        logger.info(f"Disconnected: {self.addr} (substation={self._sub_id})")
        try:
            self.conn.close()
        except Exception:
            pass
        if self._sub_id and self.disconnect_callback:
            self.disconnect_callback(self._sub_id)
