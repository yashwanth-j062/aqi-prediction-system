"""
Validation utilities
"""
from typing import Optional

def validate_aqi(aqi: Optional[float]) -> bool:
    """Validate AQI value"""
    if aqi is None:
        return False
    return 0 <= aqi <= 500

def validate_pm(pm: Optional[float]) -> bool:
    """Validate PM values"""
    if pm is None:
        return False
    return pm >= 0

def validate_temperature(temp: Optional[float]) -> bool:
    """Validate temperature"""
    if temp is None:
        return False
    return -50 <= temp <= 60  # Reasonable range for India

def validate_humidity(humidity: Optional[float]) -> bool:
    """Validate humidity"""
    if humidity is None:
        return False
    return 0 <= humidity <= 100

def validate_city_name(city_name: str) -> bool:
    """Validate city name"""
    return bool(city_name and len(city_name) >= 2)
