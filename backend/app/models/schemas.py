from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# City Schemas
class CityBase(BaseModel):
    name: str
    state: str
    latitude: float
    longitude: float

class CityResponse(CityBase):
    id: int
    is_active: bool
    
    class Config:
        from_attributes = True

# Air Quality Data Schemas
class AirQualityDataResponse(BaseModel):
    id: int
    city_id: int
    timestamp: datetime
    aqi: Optional[float] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    data_source: Optional[str] = None
    
    class Config:
        from_attributes = True

# Prediction Schemas
class PredictionResponse(BaseModel):
    id: int
    city_id: int
    predicted_aqi: float
    prediction_timestamp: datetime
    target_timestamp: datetime
    model_type: str
    predicted_category: Optional[str] = None
    confidence_score: Optional[float] = None
    
    class Config:
        from_attributes = True

# Dashboard Response Schema
class DashboardResponse(BaseModel):
    city: CityResponse
    current_aqi: Optional[float] = None
    current_category: Optional[str] = None
    health_message: Optional[str] = None
    current_data: Optional[AirQualityDataResponse] = None
    predictions: List[PredictionResponse] = []
    historical_data: List[AirQualityDataResponse] = []

# Model Metrics Schema
class ModelMetricsResponse(BaseModel):
    id: int
    model_type: str
    model_version: Optional[str] = None
    training_date: datetime
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    training_samples: Optional[int] = None
    is_active: bool
    
    class Config:
        from_attributes = True

# Health Check Schema
class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    database: str
    total_cities: int
    total_records: int
