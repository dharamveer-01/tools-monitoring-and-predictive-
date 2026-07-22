"""
start_all.py — Launcher for the Distributed AI Smart Grid Simulator.

DATA SOURCE:
  By default this script starts USB clients for S1, S2, S3.
  Each client will wait for a physical USB sensor to be plugged in.

  If you want to run a quick demo WITHOUT hardware, pass --simulate:
      python start_all.py --simulate

DISTRIBUTED SETUP (recommended):
  Run only the server + dashboard here:
      python start_all.py --server-only

  Then on each substation laptop run:
      python substations/substation_client.py --id S1 --host <SERVER_IP>
      python substations/substation_client.py --id S2 --host <SERVER_IP>
      python substations/substation_client.py --id S3 --host <SERVER_IP>
"""
import subprocess
import time
import os
import sys
import argparse

PYTHON = sys.executable


def start(command: str, name: str) -> subprocess.Popen:
    print(f"  ▶  {name}")
    return subprocess.Popen(command, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Smart Grid Simulator launcher")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use synthetic data instead of real USB sensors (demo/dev mode)",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Start only the backend + dashboard (for distributed multi-laptop setup)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="AI server IP (used when launching substation clients)",
    )
    args = parser.parse_args()

    print("⚡ Distributed AI Smart Grid Simulator\n")

    processes = []

    # ── 1. Backend ────────────────────────────────────────────────────────────
    p_backend = start(f'"{PYTHON}" api/main.py', "FastAPI Backend + AI Socket Server (port 8000 + 9999)")
    processes.append(p_backend)
    print("     Waiting for backend to initialise…")
    time.sleep(4)

    # ── 2. Substation clients ─────────────────────────────────────────────────
    if not args.server_only:
        if args.simulate:
            print("\n  ⚠️  SIMULATION MODE — using synthetic data (no real hardware)")
            p_s1 = start(
                f'"{PYTHON}" substations/substation_client.py --id S1 --host {args.host} --simulate',
                "Substation S1 [SIMULATED — healthy]",
            )
            time.sleep(0.5)
            p_s2 = start(
                f'"{PYTHON}" substations/substation_client.py --id S2 --host {args.host} --simulate --faulty',
                "Substation S2 [SIMULATED — faulty]",
            )
            time.sleep(0.5)
            p_s3 = start(
                f'"{PYTHON}" substations/substation_client.py --id S3 --host {args.host} --simulate',
                "Substation S3 [SIMULATED — healthy]",
            )
        else:
            print("\n  🔌 USB MODE — waiting for physical sensors on each COM port")
            print("     Plug in your USB sensors. Each client will auto-detect its port.")
            print("     To specify ports manually, run each client individually:")
            print(f'       python substations/substation_client.py --id S1 --port-name COM3 --host {args.host}')
            print(f'       python substations/substation_client.py --id S2 --port-name COM4 --host {args.host}')
            print(f'       python substations/substation_client.py --id S3 --port-name COM5 --host {args.host}')
            print()
            p_s1 = start(
                f'"{PYTHON}" substations/substation_client.py --id S1 --host {args.host}',
                "Substation S1 [USB — auto-detect port]",
            )
            time.sleep(0.5)
            p_s2 = start(
                f'"{PYTHON}" substations/substation_client.py --id S2 --host {args.host}',
                "Substation S2 [USB — auto-detect port]",
            )
            time.sleep(0.5)
            p_s3 = start(
                f'"{PYTHON}" substations/substation_client.py --id S3 --host {args.host}',
                "Substation S3 [USB — auto-detect port]",
            )

        processes += [p_s1, p_s2, p_s3]
        time.sleep(1)

    # ── 3. Dashboard ──────────────────────────────────────────────────────────
    # React frontend (primary)
    p_frontend = subprocess.Popen(
        'npm run dev',
        shell=True,
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'frontend'),
    )
    processes.append(p_frontend)

    # Streamlit (secondary / fallback)
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    p_dash = subprocess.Popen(
        f'"{PYTHON}" -m streamlit run dashboard/app.py --server.port 8501',
        shell=True,
        env=env,
    )
    processes.append(p_dash)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n✅ Started:")
    print("   React Dashboard → http://localhost:5173")
    print("   Streamlit (alt) → http://localhost:8501")
    print("   API docs        → http://localhost:8000/docs")
    print("   Socket          → port 9999")
    if args.server_only:
        print("\n   Waiting for substation clients to connect from other laptops…")
    print("\nPress Ctrl+C to stop everything.\n")

    try:
        p_backend.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down…")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1)
        for p in processes:
            try:
                p.kill()
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    main()
