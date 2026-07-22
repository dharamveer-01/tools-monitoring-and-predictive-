import serial.tools.list_ports

def list_available_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No USB/COM ports found. Please check if your device is plugged in.")
        return

    print("Found the following connected devices:")
    for port, desc, hwid in sorted(ports):
        print(f" - {port}: {desc}")

if __name__ == "__main__":
    list_available_ports()
