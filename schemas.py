from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TelemetryData(BaseModel):
    substation_id: str
    timestamp: datetime
    voltage: float
    current: float
    temperature: float
    harmonic_5th: float
    load_percentage: float

class HealthScore(BaseModel):
    substation_id: str
    anomaly_detected: bool
    health_score: float
    risk_level: str
    timestamp: datetime
