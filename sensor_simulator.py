import random
import time
from datetime import datetime
import json

class SensorSimulator:
    def __init__(self, sensor_id: str, sensor_type: str):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type  # e.g., 'voltage', 'current', 'thermal'
        
    def generate_reading(self, is_faulty=False):
        """Generates mock telemetry data based on the sensor type."""
        timestamp = datetime.now().isoformat()
        
        if self.sensor_type == 'voltage':
            val = random.uniform(220, 240) if not is_faulty else random.uniform(170, 190)
        elif self.sensor_type == 'current':
            val = random.uniform(10, 20) if not is_faulty else random.uniform(25, 40)
        elif self.sensor_type == 'thermal':
            val = random.uniform(50, 75) if not is_faulty else random.uniform(90, 120)
        else:
            val = random.random()
            
        return {
            "sensor_id": self.sensor_id,
            "type": self.sensor_type,
            "value": round(val, 2),
            "timestamp": timestamp,
            "is_faulty": is_faulty
        }

if __name__ == "__main__":
    # Test the standalone simulator generator
    sim = SensorSimulator("V-100", "voltage")
    print("Normal Reading:", sim.generate_reading())
    print("Faulty Reading:", sim.generate_reading(is_faulty=True))
