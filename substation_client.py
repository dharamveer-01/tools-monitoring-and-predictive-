"""
substation_client.py — Main entry point for each substation laptop.

DATA SOURCE PRIORITY (in order):
  1. USB serial device (Arduino / ESP32 / any microcontroller) — DEFAULT
  2. Android phone via ADB                                     — --source adb
  3. Simulation (synthetic data)                               — --simulate  ← EXPLICIT OPT-IN ONLY

The system NEVER falls back to simulation silently.
If no hardware is found, the client waits and tells you what to do.

Usage examples:
  # USB sensor (auto-detect port):
  python substations/substation_client.py --id S1 --host 192.168.1.100

  # USB sensor (explicit port):
  python substations/substation_client.py --id S1 --host 192.168.1.100 --port-name COM3

  # Android phone via ADB:
  python substations/substation_client.py --id S2 --host 192.168.1.100 --source adb

  # Simulation ONLY (for development/demo when no hardware is available):
  python substations/substation_client.py --id S3 --host 192.168.1.100 --simulate

  # Simulation with fault injection (for testing the AI pipeline):
  python substations/substation_client.py --id S2 --host 192.168.1.100 --simulate --faulty

  # List available USB ports:
  python substations/substation_client.py --list-ports
"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import SERVER_PORT
from shared.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Substation client — streams telemetry to the AI server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--id",          default="S1",        help="Substation ID (e.g. S1, S2, S3)")
    parser.add_argument("--host",        default="localhost",  help="AI server IP address")
    parser.add_argument("--server-port", default=SERVER_PORT,  type=int, help="AI server TCP port")

    # Data source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        choices=["usb", "adb"],
        default="usb",
        help="Hardware data source: 'usb' (default) or 'adb' (Android phone)",
    )
    source_group.add_argument(
        "--simulate",
        action="store_true",
        help="Use SYNTHETIC data instead of real hardware. Only for development/demo.",
    )

    # USB options
    parser.add_argument("--port-name", default=None,  help="COM port (e.g. COM3, /dev/ttyUSB0). Auto-detects if omitted.")
    parser.add_argument("--baud",      default=9600,  type=int, help="Serial baud rate (default: 9600)")

    # Simulation options (only relevant with --simulate)
    parser.add_argument("--faulty",     action="store_true", help="[--simulate only] Enable fault injection")
    parser.add_argument("--fault-prob", default=0.3,         type=float, help="[--simulate only] Fault probability 0–1")

    # Utility
    parser.add_argument("--list-ports", action="store_true", help="List available USB/COM ports and exit")

    args = parser.parse_args()

    # ── List ports and exit ───────────────────────────────────────────────────
    if args.list_ports:
        from substations.usb_substation_client import print_available_ports
        print_available_ports()
        sys.exit(0)

    # ── Route to the correct data source ─────────────────────────────────────

    if args.simulate:
        # Explicit simulation mode
        logger.warning(
            f"[{args.id}] *** SIMULATION MODE *** "
            "No real hardware will be used. Pass --source usb or --source adb for real data."
        )
        _run_simulation(args)

    elif args.source == "adb":
        logger.info(f"[{args.id}] Starting ADB (Android phone) client.")
        _run_adb(args)

    else:
        # Default: USB serial
        logger.info(f"[{args.id}] Starting USB serial client.")
        _run_usb(args)


# ── Runners ───────────────────────────────────────────────────────────────────

def _run_usb(args) -> None:
    from substations.usb_substation_client import UsbSubstationClient
    client = UsbSubstationClient(
        substation_id=args.id,
        host=args.host,
        server_port=args.server_port,
        port_name=args.port_name,
        baud_rate=args.baud,
    )
    client.start_streaming()


def _run_adb(args) -> None:
    from substations.adb_substation_client import AdbSubstationClient
    client = AdbSubstationClient(
        substation_id=args.id,
        host=args.host,
        port=args.server_port,
    )
    client.start_streaming()


def _run_simulation(args) -> None:
    """
    Synthetic data mode — only used when --simulate is explicitly passed.
    Useful for development, demos, and testing the AI pipeline without hardware.
    """
    import json
    import time
    import socket
    from substations.telemetry_generator import TelemetryGenerator
    from substations.fault_simulator import FaultSimulator
    from shared.config import TELEMETRY_INTERVAL

    gen = TelemetryGenerator(args.id)
    fault_sim = FaultSimulator(fault_probability=args.fault_prob) if args.faulty else None

    # Connect to server
    sock = None
    def connect():
        nonlocal sock
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((args.host, args.server_port))
                logger.info(f"[{args.id}] [SIM] Connected to {args.host}:{args.server_port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(f"[{args.id}] [SIM] Cannot connect: {e}. Retrying in 2s…")
                time.sleep(2)

    connect()
    try:
        while True:
            packet = gen.generate()
            if fault_sim:
                packet = fault_sim.maybe_inject(packet)
            message = json.dumps(packet) + "\n"
            try:
                sock.sendall(message.encode("utf-8"))
                fault_tag = f" [FAULT:{packet.get('fault_type')}]" if packet.get("fault_type") else ""
                logger.info(
                    f"[{args.id}] [SIM] V={packet['voltage']}V "
                    f"T={packet['temperature']}°C "
                    f"Load={packet['load_percentage']}%{fault_tag}"
                )
            except (BrokenPipeError, OSError) as e:
                logger.warning(f"[{args.id}] [SIM] Send failed: {e} — reconnecting…")
                try:
                    sock.close()
                except Exception:
                    pass
                connect()
            time.sleep(TELEMETRY_INTERVAL)
    except KeyboardInterrupt:
        logger.info(f"[{args.id}] [SIM] Stopped.")
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
